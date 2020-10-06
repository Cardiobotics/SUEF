import torch
import torch.nn as nn
from utils import AverageMeter
from sklearn.metrics import r2_score
import time
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import os
import neptune
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf


def train_and_validate(model, train_data_loader, train_sampler,  val_data_loader, cfg, experiment=None):

    # Set visible devices
    parallel_model = cfg.performance.parallel_mode
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Set cuda
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)

    # CUDNN Auto-tuner. Use True when input size and model is static
    torch.backends.cudnn.benchmark = cfg.performance.cuddn_auto_tuner

    # Set loss criterion
    criterion = nn.MSELoss(reduction='none')

    # Set optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
    if parallel_model:
        print("Available GPUS: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    use_half_prec = cfg.performance.half_precision

    # Initialize GradScaler for autocasting
    scaler = GradScaler(enabled=use_half_prec)

    print('Model parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_buffs  # in bytes
    print('Model memory size: {}'.format(mem))

    # Initialize scheduler
    use_scheduler = cfg.training.use_scheduler
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.training.sched_step_size,
                                                    gamma=cfg.training.sched_gamma)

    # Maximum value used for gradient clipping = max fp16/2
    gradient_clipping = cfg.performance.gradient_clipping
    max_norm = cfg.performance.gradient_clipping_max_norm

    # Set anomaly detection
    torch.autograd.set_detect_anomaly(cfg.performance.anomaly_detection)

    # Begin training

    max_val_r2 = 0

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
        for j, (inputs_t, targets_t, indexes_t) in enumerate(train_data_loader):
            # Update timer for data retrieval
            data_time_t.update(time.time() - end_time_t)
            
            # Move input to CUDA if available
            if cuda_available:
                targets_t = targets_t.to(device, non_blocking=True)
                inputs_t = inputs_t.to(device, non_blocking=True)

            # Do forward and backwards pass

            # Get model train output and train loss
            with autocast(enabled=use_half_prec):
                outputs_t, input_vectors, sequenceOut, maskSample = model(inputs_t)
                loss_t = criterion(outputs_t, targets_t)
                loss_mean_t = loss_t.mean()
            if cfg.data_loader.weighted_sampler:
                for index, loss in zip(indexes_t, loss_t.cpu().detach()):
                    loss_ratio = loss/loss_mean_t.cpu().detach()
                    train_sampler.weights[index] = loss_ratio
            # Backwards pass and step
            optimizer.zero_grad()

            # Backwards pass
            scaler.scale(loss_mean_t).backward()
            #loss_mean_t.backward()

            # Gradient Clipping
            if gradient_clipping:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), max_norm)

            scaler.step(optimizer)
            #optimizer.step()
            scaler.update()

            # Update metrics
            r2_targets_t = targets_t.cpu().detach()
            r2_outputs_t = outputs_t.cpu().detach()
            r2_t = r2_score(r2_targets_t, r2_outputs_t)
            r2_values_t.update(r2_t)
            losses_t.update(loss_mean_t)

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
        end_time_v = time.time()
        if use_scheduler:
            scheduler.step()


        # Validation
        model.eval()
        for k, (inputs_v, targets_v, _) in enumerate(val_data_loader):
            # Update timer for data retrieval
            data_time_v.update(time.time() - end_time_v)

            # Move input to CUDA if available
            if cuda_available:
                targets_v = targets_v.to(device, non_blocking=True)
                inputs_v = inputs_v.to(device, non_blocking=True)
            with torch.no_grad():
            # Get model validation output and validation loss
                with autocast(enabled=use_half_prec):
                    outputs_v, _, _, _ = model(inputs_v)
                    loss_v = criterion(outputs_v, targets_v)
                    loss_mean_v = loss_v.mean()

            # Update metrics
            r2_targets_v = targets_v.cpu().detach()
            r2_outputs_v = outputs_v.cpu().detach()
            r2_v = r2_score(r2_targets_v, r2_outputs_v)
            r2_values_v.update(r2_v)
            losses_v.update(loss_mean_v)

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
        print('Example targets: {} \n Example outputs: {}'.format(torch.squeeze(targets_v), torch.squeeze(outputs_v)))

        if cfg.logging.logging_enabled:
            log_metrics(experiment, losses_v.avg, r2_values_v.avg)

        if cfg.training.checkpointing_enabled and cfg.logging.logging_enabled:
            if r2_values_v.avg > max_val_r2:
                checkpoint_name = cfg.training.checkpoint_save_path + cfg.model.name + '_' + cfg.data.type + '_'\
                                  + cfg.data.name + '_exp_' + experiment.id + '.pth'
                save_checkpoint(checkpoint_name, model, optimizer)
                max_val_r2 = r2_values_v.avg
        elif cfg.training.checkpointing_enabled:
            if r2_values_v.avg > max_val_r2:
                checkpoint_name = cfg.training.checkpoint_save_path + cfg.model.name + '_' + cfg.data.type + '_'\
                                  + cfg.data.name + '_test' + '.pth'
                save_checkpoint(checkpoint_name, model, optimizer)
                max_val_r2 = r2_values_v.avg


def log_metrics(experiment, loss, r2):
    experiment.log_metric('loss', loss)
    experiment.log_metric('r2', r2)


def save_checkpoint(save_file_path, model, optimizer):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(save_states, save_file_path)


def restore_checkpoint(args, model, optimizer, scheduler, checkpoint_path):
    # Restore
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, scheduler
