import torch
import torch.nn as nn
from utils.utils import AverageMeter
from sklearn.metrics import r2_score
import time
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import os
import neptune


def train_and_validate(img_model, flow_model, train_data_loader, val_data_loader, cfg):

    # Set visible devices
    parallel_model = cfg.performance.parallel_mode
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Set cuda
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        # torch.cuda.empty_cache()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    img_model.to(device)
    flow_model.to(device)

    # CUDNN Auto-tuner. Use True when input size and model is static
    torch.backends.cudnn.benchmark = cfg.performance.cuddn_auto_tuner

    if cfg.training.freeze_lower:
        for p in img_model.parameters():
            p.requires_grad = False
        img_model.logits.conv3d.weight.requires_grad = True
        img_model.logits.conv3d.bias.requires_grad = True
        for p in flow_model.parameters():
            p.requires_grad = False
        flow_model.logits.conv3d.weight.requires_grad = True
        flow_model.logits.conv3d.bias.requires_grad = True

    # Set loss criterion
    criterion = nn.MSELoss()

    # Set optimizer
    img_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, img_model.parameters()),
                                      lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
    flow_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, flow_model.parameters()),
                                       lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
    if parallel_model:
        print("Available GPUS: {}".format(torch.cuda.device_count()))
        img_model = nn.DataParallel(img_model)
        flow_model = nn.DataParallel(flow_model)

    use_half_prec = cfg.performance.half_precision

    # Initialize GradScaler for autocasting
    scaler = GradScaler(enabled=use_half_prec)

    print('Img Model parameters: {}'.format(sum(p.numel() for p in img_model.parameters() if p.requires_grad)))
    print('Flow Model parameters: {}'.format(sum(p.numel() for p in flow_model.parameters() if p.requires_grad)))

    mem_params = sum([param.nelement() * param.element_size() for param in img_model.parameters()])
    mem_buffs = sum([buf.nelement() * buf.element_size() for buf in img_model.buffers()])
    mem = mem_params + mem_buffs  # in bytes
    print('Img Model memory size: {}'.format(mem))

    mem_params = sum([param.nelement() * param.element_size() for param in flow_model.parameters()])
    mem_buffs = sum([buf.nelement() * buf.element_size() for buf in flow_model.buffers()])
    mem = mem_params + mem_buffs  # in bytes
    print('Img Model memory size: {}'.format(mem))

    # Initialize scheduler
    use_scheduler = cfg.training.use_scheduler
    if use_scheduler:
        img_scheduler = torch.optim.lr_scheduler.StepLR(img_optimizer, step_size=cfg.training.sched_step_size,
                                                        gamma=cfg.training.sched_gamma)
        flow_scheduler = torch.optim.lr_scheduler.StepLR(flow_optimizer, step_size=cfg.training.sched_step_size,
                                                         gamma=cfg.training.sched_gamma)

    # Maximum value used for gradient clipping = max fp16/2
    gradient_clipping = cfg.performance.gradient_clipping
    max_norm = cfg.performance.gradient_clipping_max_norm

    # Set anomaly detection
    torch.autograd.set_detect_anomaly(cfg.performance.anomaly_detection)

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
        img_model.train()
        flow_model.train()
        for j, (img_inputs_t, flow_inputs_t, targets_t, index, uid) in enumerate(train_data_loader):
            # Update timer for data retrieval
            data_time_t.update(time.time() - end_time_t)

            # Move input to CUDA if available
            if cuda_available:
                targets_t = targets_t.to(device, non_blocking=True)
                img_inputs_t = img_inputs_t.to(device, non_blocking=True)
                flow_inputs_t = flow_inputs_t.to(device, non_blocking=True)

            # Do forward and backwards pass

            # Get model train output and train loss
            with autocast(enabled=use_half_prec):
                img_outputs_t = img_model(img_inputs_t)
                flow_outputs_t = flow_model(flow_inputs_t)
                outputs_t = (img_outputs_t + flow_outputs_t) / 2
                loss_t = criterion(outputs_t, targets_t)

            # Backwards pass and step
            img_optimizer.zero_grad()
            flow_optimizer.zero_grad()

            # Backwards pass
            scaler.scale(loss_t).backward()

            # Gradient Clipping
            if gradient_clipping:
                scaler.unscale_(img_optimizer)
                scaler.unscale_(flow_optimizer)
                torch.nn.utils.clip_grad_value_(img_model.parameters(), max_norm)
                torch.nn.utils.clip_grad_value_(flow_model.parameters(), max_norm)

            scaler.step(img_optimizer)
            scaler.step(flow_optimizer)
            scaler.update()

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
                      .format(j + 1, len(train_data_loader), i + 1, batch_time=batch_time_t, data_time=data_time_t,
                              loss=losses_t, r2=r2_values_t))

            # Reset end timer
            end_time_t = time.time()

        # End of training epoch prints and updates
        print('Finished Training Epoch: {} \t '
              'Training Time: {batch_time.avg:.3f} \t '
              'Training Data Time: {data_time.avg:.3f}) \t '
              'Training Loss: {loss.avg:.4f} \t '
              'Training R2 score: {r2.avg:.3f} \t'
              .format(i + 1, batch_time=batch_time_t, data_time=data_time_t, loss=losses_t, r2=r2_values_t))
        end_time_v = time.time()
        if use_scheduler:
            img_scheduler.step()
            flow_scheduler.step()

        # Validation
        img_model.eval()
        flow_model.eval()
        for k, (img_inputs_v, flow_inputs_v, targets_v) in enumerate(val_data_loader):
            # Update timer for data retrieval
            data_time_v.update(time.time() - end_time_v)

            # Move input to CUDA if available
            if cuda_available:
                targets_v = targets_v.to(device, non_blocking=True)
                img_inputs_v = img_inputs_v.to(device, non_blocking=True)
                flow_inputs_v = flow_inputs_v.to(device, non_blocking=True)
            with torch.no_grad():
                # Get model validation output and validation loss
                with autocast(enabled=use_half_prec):
                    img_outputs_v = img_model(img_inputs_v)
                    flow_outputs_v = flow_model(flow_inputs_v)
                    outputs_v = (img_outputs_v + flow_outputs_v) / 2
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
              .format(i + 1, batch_time=batch_time_v, data_time=data_time_v, loss=losses_v, r2=r2_values_v))
        print('Example targets: {} \n Example outputs: {}'.format(torch.squeeze(targets_v), torch.squeeze(outputs_v)))

        if cfg.logging.logging_enabled:
            log_metrics(losses_v.avg, r2_values_v.avg)

    if cfg.training.checkpointing_enabled:
        img_checkpoint_name = cfg.training.checkpoint_save_path + cfg.model.name + '_' + cfg.data_stream.type + '_' + cfg.data.name + '_img.pth'
        flow_checkpoint_name = cfg.training.checkpoint_save_path + cfg.model.name + '_' + cfg.data_stream.type + '_' + cfg.data.name + '_flow.pth'
        save_checkpoint(img_checkpoint_name, img_model, img_optimizer, img_scheduler)
        save_checkpoint(flow_checkpoint_name, flow_model, flow_optimizer, flow_scheduler)


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
    }
    torch.save(save_states, save_file_path)


def restore_checkpoint(args, model, optimizer, scheduler, checkpoint_path):
    # Restore
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, scheduler
