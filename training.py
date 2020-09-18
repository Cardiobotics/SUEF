import dcm_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import autograd
from utils import AverageMeter
from sklearn.metrics import r2_score
import time
from apex import amp
from apex.optimizers import FusedAdam
import os
import neptune
import hydra
from omegaconf import DictConfig, OmegaConf


def train_and_validate(model, cfg):

    # Create DataLoaders for training and validation
    train_d_set = dcm_dataset.DCMDataset(cfg.data.train_view_path, cfg.data.train_target_path, cfg.transforms,
                                         cfg.augmentations, cfg.data.file_sep)
    train_sampler = RandomSampler(train_d_set)
    train_data_loader = DataLoader(train_d_set, sampler=train_sampler, batch_size=cfg.data_loader.batch_size,
                                   num_workers=cfg.data_loader.n_workers)

    val_d_set = dcm_dataset.DCMDataset(cfg.data.val_view_path, cfg.data.val_target_path, cfg.transforms,
                                       cfg.augmentations, cfg.data.file_sep)
    test_sampler = SequentialSampler(val_d_set)
    val_data_loader = DataLoader(val_d_set, sampler=test_sampler, batch_size=cfg.data_loader.batch_size,
                                 num_workers=cfg.data_loader.n_workers)

    # Set cuda and clear cache
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        #torch.cuda.empty_cache()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # CUDNN Auto-tuner. Use True when input size and model is static
    torch.backends.cudnn.benchmark = cfg.performance.cuddn_auto_tuner

    # Set Apex optimization level
    # 00 = no optimization
    # 01 = Patched Tensor methods/PyTorch functions, no mixed precision for weights
    # 02 = Mixed precision for model weights, batchnorm as FP32, no patch
    # 03 = Full FP16 everywhere
    opt_level = cfg.performance.amp_opt_level
    amp_enabled = cfg.performance.amp_enabled

    # Set parallelization, optimizer and loss function
    parallel_model = cfg.performance.parallel_mode
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if parallel_model:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        model = nn.DataParallel(model)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model.to(device)
    print('Model parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if cfg.training.freeze_lower:
        for p in model.parameters():
            p.requires_grad = False
        model.logits.conv3d.weight.requires_grad = True
        model.logits.conv3d.bias.requires_grad = True


    # Set loss criterion
    criterion = nn.MSELoss()

    # Set optimizer
    if cuda_available and amp_enabled:
        optimizer = FusedAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.optimizer.learning_rate,
                              weight_decay=cfg.optimizer.weight_decay)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)

    # Maximum value used for gradient clipping = max fp16/2
    gradient_clipping = cfg.performance.gradient_clipping
    max_norm = cfg.performance.gradient_clipping_max_norm

    # Set anomaly detection
    torch.autograd.set_detect_anomaly(cfg.performance.anomaly_detection)

    # Put model on Apex Amp
    if amp_enabled:
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_buffs  # in bytes
    print('Model memory size: {}'.format(mem))

    # Initialize scheduler
    use_scheduler = cfg.training.use_scheduler
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.training.sched_step_size,
                                                    gamma=cfg.training.sched_gamma)

    # Begin training

    for i in range(cfg.training.epochs):

        batch_time_t = AverageMeter()
        data_time_t = AverageMeter()
        losses_t = AverageMeter()
        r2_values_t = AverageMeter()

        batch_time_v = AverageMeter()
        data_time_v = AverageMeter()
        losses_v = AverageMeter()
        r2_values_v = AverageMeter()

        end_time_t = time.time()
        # Training
        model.train()
        for j, (inputs_t, targets_t) in enumerate(train_data_loader):
            # Update timer for data retrieval
            data_time_t.update(time.time() - end_time_t)
            
            # Move input to CUDA if available
            if cuda_available:
                targets_t = targets_t.to(device, non_blocking=True)
                inputs_t = inputs_t.to(device, non_blocking=True)

            # Do forward and backwards pass

            # Get model train output and train loss
            outputs_t = model(inputs_t.float())
            loss_t = criterion(outputs_t, targets_t)

            # Backwards pass and step
            optimizer.zero_grad()
            # Backwards pass
            if amp_enabled:
                with amp.scale_loss(loss_t, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_t.backward()
            optimizer.step()

            # Gradient Clipping
            if gradient_clipping:
                torch.nn.utils.clip_grad_value_(amp.master_params(optimizer), max_norm)

            # Update metrics
            r2_targets_t = targets_t.cpu().detach()
            r2_outputs_t = outputs_t.cpu().detach()
            r2_t = r2_score(r2_targets_t, r2_outputs_t)
            r2_values_t.update(r2_t)
            losses_t.update(loss_t)

            # Update timer for batch
            batch_time_t.update(time.time() - end_time_t)

            if j % 10 == 0:
                print('Training Batch: [{}/{}] in epoch: {} \t '
                      'Training Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) \t '
                      'Training Data Time: {data_time.val:.3f} ({data_time.avg:.3f}) \t '
                      'Training Loss: {loss.val:.4f} ({loss.avg:.4f}) \t '
                      'Training R2 Score: {r2.val:.3f} ({r2.avg:.3f}) \t'
                      .format(j+1, len(train_data_loader), i + 1, batch_time=batch_time_t, data_time=data_time_t,
                              loss=losses_t, r2=r2_values_t))

            # Reset end timer
            end_time_t = time.time()
        
        # End of training epoch prints and updates
        print('Finished Training Epoch: {} \t '
              'Training Time: {batch_time.avg:.3f} \t '
              'Training Data Time: {data_time.avg:.3f}) \t '
              'Training Loss: {loss.avg:.4f} \t '
              'Training R2 score: {r2.avg:.3f} \t'
              .format(i+1, batch_time=batch_time_t, data_time=data_time_t, loss=losses_t, r2=r2_values_t))
        print('Example targets: {} \n Example outputs: {}'.format(targets_t, outputs_t))
        end_time_v = time.time()
        if use_scheduler:
            scheduler.step()

        if cfg.training.freeze_lower and j == 3:
            for p in model.parameters():
                p.requires_grad = True
            if cuda_available and amp_enabled:
                optimizer = FusedAdam(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=cfg.optimizer.learning_rate,
                                      weight_decay=cfg.optimizer.weight_decay)
            else:
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                             lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)


        # Validation
        model.eval()
        with torch.no_grad():
            for k, (inputs_v, targets_v) in enumerate(val_data_loader):
                # Update timer for data retrieval
                data_time_v.update(time.time() - end_time_v)

                # Move input to CUDA if available
                if cuda_available:
                    targets_v = targets_v.to(device, non_blocking=True)
                    inputs_v = inputs_v.to(device, non_blocking=True)

                # Get model validation output and validation loss
                outputs_v = model(inputs_v)
                loss_v = criterion(outputs_v, targets_v)

                # Update metrics
                r2_targets_v = targets_v.cpu().detach()
                r2_outputs_v = outputs_v.cpu().detach()
                r2_v = r2_score(r2_targets_v, r2_outputs_v)
                r2_values_v.update(r2_v)
                losses_v.update(loss_v)

                # Update timer for batch
                batch_time_v.update(time.time() - end_time_v)
                end_time_v = time.time()

                if k % 10 == 0:
                    print('Validation Batch: [{}/{}] in epoch: {} \t '
                          'Validation Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) \t '
                          'Validation Data Time: {data_time.val:.3f} ({data_time.avg:.3f}) \t '
                          'Validation Loss: {loss.val:.4f} ({loss.avg:.4f}) \t '
                          'Validation R2 Score: {r2.val:.3f} ({r2.avg:.3f}) \t'
                          .format(k + 1, len(val_data_loader), i + 1, batch_time=batch_time_v, data_time=data_time_v,
                                  loss=losses_v, r2=r2_values_v))

        # End of validation epoch prints and updates
        print('Finished Validation Epoch: {} \t '
              'Validation Time: {batch_time.avg:.3f} \t '
              'Validation Data Time: {data_time.avg:.3f} \t '
              'Validation Loss: {loss.avg:.4f} \t '
              'Validation R2 score: {r2.avg:.3f} \t'
              .format(i+1, batch_time=batch_time_v, data_time=data_time_v, loss=losses_v, r2=r2_values_v))

        if cfg.training.checkpointing_enabled:
            checkpoint_name = cfg.training.checkpoint_save_path + cfg.model.name + '.pth'
            save_checkpoint(checkpoint_name, model, optimizer, scheduler)

        if cfg.logging.logging_enabled:
            log_metrics(losses_v.avg, r2_values_v.avg)


def log_metrics(loss, r2):
    neptune.log_metric('loss', loss)
    neptune.log_metric('r2', r2)


def save_checkpoint(save_file_path, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'amp': amp.state_dict()
    }
    torch.save(save_states, save_file_path)


def restore_checkpoint(args, model, optimizer, scheduler, checkpoint_path):
    # Restore
    checkpoint = torch.load(checkpoint_path)

    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, scheduler