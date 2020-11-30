import torch
import torch.nn as nn
from utils.utils import AverageMeter
from sklearn.metrics import r2_score
import time
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import os
import numpy as np
import pandas as pd


def train_and_validate(model, train_data_loader, val_data_loader, cfg, experiment=None):

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
    use_scheduler = cfg.optimizer.use_scheduler
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.optimizer.s_patience, factor=cfg.optimizer.s_factor)

    # Maximum value used for gradient clipping = max fp16/2
    gradient_clipping = cfg.performance.gradient_clipping
    max_norm = cfg.performance.gradient_clipping_max_norm

    # Set anomaly detection
    torch.autograd.set_detect_anomaly(cfg.performance.anomaly_detection)

    # Begin training

    max_val_r2 = -10000

    for i in range(cfg.training.epochs):

        start_time_epoch = time.time()

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
        for j, (inputs_t, targets_t, indexes_t, _) in enumerate(train_data_loader):
            # Update timer for data retrieval
            data_time_t.update(time.time() - end_time_t)
            
            # Move input to CUDA if available
            if cuda_available:
                if len(inputs_t) > 1:
                    for p, inp in enumerate(inputs_t):
                        inputs_t[p] = inp.to(device, non_blocking=True)
                else:
                    inputs_t = inputs_t.to(device, non_blocking=True)
                targets_t = targets_t.to(device, non_blocking=True)

            # Do forward and backwards pass

            # Get model train output and train loss
            with autocast(enabled=use_half_prec):
                outputs_t = model(inputs_t)
                loss_t = criterion(outputs_t, targets_t)
                loss_mean_t = loss_t.mean()
            if cfg.data_loader.weighted_sampler:
                for index, loss in zip(indexes_t, loss_t.cpu().detach()):
                    loss_ratio = loss/loss_mean_t.cpu().detach()
                    loss_ratio = torch.clamp(loss_ratio, min=0.1, max=3)
                    train_data_loader.sampler.weights[index] = loss_ratio
            # Backwards pass and step
            optimizer.zero_grad()

            # Backwards pass
            scaler.scale(loss_mean_t).backward()

            # Gradient Clipping
            if gradient_clipping:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()

            # Update metrics
            try:
                is_finite = torch.isfinite(r2_outputs_t).all()
                if not is_finite:
                    raise ValueError('Output from model not finite')
                r2_targets_t = targets_t.cpu().detach()
                r2_outputs_t = outputs_t.cpu().detach()
                r2_t = r2_score(r2_targets_t, r2_outputs_t)
                r2_values_t.update(r2_t)
                losses_t.update(loss_mean_t)
            except ValueError as ve:
                print('Failed to calculate R2 with error: {} and output: {}'.format(ve, r2_outputs_t))

            # Update timer for batch
            batch_time_t.update(time.time() - end_time_t)

            if j % 100 == 0:
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

        # Validation

        end_time_v = time.time()
        model.eval()
        all_out_v = np.zeros((0))
        all_target_v = np.zeros((0))
        all_uids_v = np.zeros((0))
        all_loss_v = np.zeros((0))
        for k, (inputs_v, targets_v, _, uids_v) in enumerate(val_data_loader):
            # Update timer for data retrieval
            data_time_v.update(time.time() - end_time_v)

            # Move input to CUDA if available
            if cuda_available:
                if len(inputs_v) > 1:
                    for p, inp in enumerate(inputs_v):
                        inputs_v[p] = inp.to(device, non_blocking=True)
                else:
                    inputs_v = inputs_v.to(device, non_blocking=True)
                targets_v = targets_v.to(device, non_blocking=True)
            with torch.no_grad():
                # Get model validation output and validation loss
                with autocast(enabled=use_half_prec):
                    outputs_v = model(inputs_v)
                    loss_v = criterion(outputs_v, targets_v)
                    loss_mean_v = loss_v.mean()

            # Update timer for batch
            batch_time_v.update(time.time() - end_time_v)

            # Update metrics
            if cfg.evaluation.use_best_sample:
                all_out_v = np.concatenate((all_out_v, outputs_v.squeeze(axis=1).cpu().detach().numpy()))
                all_target_v = np.concatenate((all_target_v, targets_v.squeeze(axis=1).cpu().detach().numpy()))
                all_uids_v = np.concatenate((all_uids_v, uids_v))
                all_loss_v = np.concatenate((all_loss_v, loss_v.cpu().squeeze(axis=1).detach().numpy()))
            else:
                r2_targets_v = targets_v.cpu().detach().numpy()
                r2_outputs_v = outputs_v.cpu().detach()
                r2_v = r2_score(r2_targets_v, r2_outputs_v)
                r2_values_v.update(r2_v)
                losses_v.update(loss_mean_v)

                if k % 100 == 0:
                    print('Validation Batch: [{}/{}] in epoch: {} \t '
                          'Validation Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) \t '
                          'Validation Data Time: {data_time.val:.3f} ({data_time.avg:.3f}) \t '
                          'Validation Loss: {loss.val:.4f} ({loss.avg:.4f}) \t '
                          'Validation R2 Score: {r2.val:.3f} ({r2.avg:.3f}) \t'
                          .format(k + 1, len(val_data_loader), i + 1, batch_time=batch_time_v, data_time=data_time_v,
                                  loss=losses_v, r2=r2_values_v))

            end_time_v = time.time()

        if cfg.evaluation.use_best_sample:
            # Only use validation results from examination with the lowest loss
            val_data = np.array((all_uids_v, all_out_v, all_target_v, all_loss_v))
            val_data = val_data.transpose(1, 0)
            pd_val_data = pd.DataFrame(val_data, columns=['us_id', 'output', 'target', 'loss'])
            pd_val_data[['output', 'target', 'loss']] = pd_val_data[['output', 'target', 'loss']].astype(np.float32)
            val_ue = pd_val_data.drop_duplicates(subset='us_id')[['us_id', 'target']]
            all_mean_loss = []
            for ue in val_ue.itertuples():
                exam_results = pd_val_data[pd_val_data['us_id'] == ue.us_id]
                num_combinations = len(exam_results)
                weight = 1/num_combinations
                mean_exam_loss = exam_results['loss'].mean()
                all_mean_loss.append(mean_exam_loss)
                for indx in exam_results.index:
                    pd_val_data.loc[indx, 'r2_weight'] = weight
            np_loss = np.array(all_mean_loss, dtype=np.float32)
            v_loss_mean = np_loss.mean()
            targets = pd_val_data['target'].to_numpy()
            outputs = pd_val_data['output'].to_numpy()
            weights = pd_val_data['r2_weight'].to_numpy()
            v_r2 = r2_score(targets, outputs, sample_weight=weights)
        else:
            v_loss_mean = losses_v.avg
            v_r2 = r2_values_v.avg

        # End of validation epoch prints and updates
        print('Finished Validation Epoch: {} \t '
              'Validation Time: {batch_time.avg:.3f} \t '
              'Validation Data Time: {data_time.avg:.3f} \t '
              'Validation Loss: {loss:.4f} \t '
              'Validation R2 score: {r2:.3f} \t'
              .format(i+1, batch_time=batch_time_v, data_time=data_time_v, loss=v_loss_mean, r2=v_r2))
        print('Example targets: {} \n Example outputs: {}'.format(torch.squeeze(targets_v), torch.squeeze(outputs_v)))

        if use_scheduler:
            scheduler.step(v_loss_mean)

        if cfg.training.checkpointing_enabled and cfg.logging.logging_enabled:
            if v_r2 > max_val_r2:
                checkpoint_name = cfg.training.checkpoint_save_path + cfg.model.name + '_' + cfg.data.type + '_'\
                                  + cfg.data.name + '_exp_' + experiment.id + '.pth'
                save_checkpoint(checkpoint_name, model, optimizer)
                max_val_r2 = v_r2
        elif cfg.training.checkpointing_enabled:
            if v_r2 > max_val_r2:
                checkpoint_name = cfg.training.checkpoint_save_path + cfg.model.name + '_' + cfg.data.type + '_'\
                                  + cfg.data.name + '_test' + '.pth'
                save_checkpoint(checkpoint_name, model, optimizer)
                max_val_r2 = v_r2

        if cfg.logging.logging_enabled:
            log_metrics(experiment, losses_t.avg, r2_values_t.avg, v_loss_mean, v_r2, max_val_r2)

        epoch_time = time.time() - start_time_epoch
        rem_epochs = cfg.training.epochs - (i+1)
        rem_time = rem_epochs * epoch_time
        print('Epoch {} completed. Time to complete: {}. Estimated remaining time: {}'.format(i+1, epoch_time, format_time(rem_time)))


def log_metrics(experiment, t_loss, t_r2, v_loss, v_r2, best_v_r2):
    experiment.log_metric('training_loss', t_loss)
    experiment.log_metric('training_r2', t_r2)
    experiment.log_metric('validation_loss', v_loss)
    experiment.log_metric('validation_r2', v_r2)
    experiment.log_metric('best_val_r2', best_v_r2)


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

def format_time(time_as_seconds):
    s = time.localtime(time_as_seconds)
    seconds = s.tm_sec
    minutes = s.tm_min
    hours = s.tm_hour
    days = s.tm_yday - 1
    return '{} days, {} hours, {} minutes and {} seconds'.format(days, hours, minutes, seconds)
