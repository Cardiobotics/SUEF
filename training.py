import torch
import torch.nn as nn
from utils.utils import AverageMeter
from sklearn.metrics import r2_score, accuracy_score, top_k_accuracy_score
import time
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from hinge_loss import HingeLossRegression
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

    if cfg.model.n_classes > 1:
        metric_name = 'Accuracy'
        goal_type = 'classification'
    else:
        metric_name = 'R2'
        goal_type = 'regression'

    if cfg.training.freeze_lower:
        for p in model.parameters():
            p.requires_grad = False
        model.Linear_layer.weight.requires_grad = True
        model.Linear_layer.bias.requires_grad = True

    # Create Criterion and Optimizer
    if cfg.optimizer.loss_function == 'hinge':
        # Set loss criterion
        criterion = HingeLossRegression(cfg.optimizer.loss_epsilon, reduction=None)
        # Hinge loss is dependent on L2 regularization so we cannot use AdamW
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.loss_function == 'mse':
        # Set loss criterion
        criterion = nn.MSELoss(reduction='none')
        # Set optimizer
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.loss_function == 'cross-entropy':
        # Get counts for each class
        # Instantiate class counts to 1 instead of 0 to prevent division by zero in case data is missing
        class_counts = np.array(cfg.model.n_classes*[1])
        for i in train_data_loader.dataset.unique_exams['target'].value_counts().index:
            class_counts[i] = train_data_loader.dataset.unique_exams['target'].value_counts().loc[i]
        # Calculate the inverse normalized ratio for each class
        weights = class_counts / class_counts.sum()
        weights = 1 / weights
        weights = weights / weights.sum()
        weights = torch.FloatTensor(weights).cuda()
        criterion = nn.CrossEntropyLoss(weight=weights, reduction='none')
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
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=len(train_data_loader))
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,150,350], gamma=0.1)

    # Maximum value used for gradient clipping = max fp16/2
    gradient_clipping = cfg.performance.gradient_clipping
    max_norm = cfg.performance.gradient_clipping_max_norm

    # Set anomaly detection
    torch.autograd.set_detect_anomaly(cfg.performance.anomaly_detection)

    # Begin training

    max_val_metric = -10000

    for i in range(cfg.training.epochs):

        start_time_epoch = time.time()

        batch_time_t = AverageMeter()
        data_time_t = AverageMeter()
        losses_t = AverageMeter()
        metric_values_t = AverageMeter()

        batch_time_v = AverageMeter()
        data_time_v = AverageMeter()
        losses_v = AverageMeter()
        metric_values_v = AverageMeter()
        if goal_type == 'classification':
            top3_values_v = AverageMeter()
            top5_values_v = AverageMeter()

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
                        if not torch.isfinite(inp).all():
                            raise ValueError('Input from dataloader not finite')
                        inputs_t[p] = inp.to(device, non_blocking=True)
                else:
                    if not torch.isfinite(inputs_t).all():
                        raise ValueError('Input from dataloader not finite')
                    inputs_t = inputs_t.to(device, non_blocking=True)
                if goal_type == 'classification':
                    targets_t = targets_t.long().squeeze()
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

            # Calculate and update metrics
            try:
                if not torch.isfinite(outputs_t).all():
                    raise ValueError('Output from model not finite')
                metric_targets_t = targets_t.cpu().detach().numpy()
                metric_outputs_t = outputs_t.cpu().detach().numpy()
                if goal_type == 'regression':
                    metric_t = r2_score(metric_targets_t, metric_outputs_t)
                else:
                    predictions_t = np.argmax(metric_outputs_t, 1)
                    metric_t = accuracy_score(metric_targets_t, predictions_t)
                metric_values_t.update(metric_t)
                losses_t.update(loss_mean_t)
            except ValueError as ve:
                print('Failed to calculate {} with error: {} and output: {}'.format(metric_name, ve, outputs_t))

            # Update timer for batch
            batch_time_t.update(time.time() - end_time_t)

            if j % 100 == 0:
                print('Training Batch: [{}/{}] in epoch: {} \t '
                      'Training Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) \t '
                      'Training Data Time: {data_time.val:.3f} ({data_time.avg:.3f}) \t '
                      'Training Loss: {loss.val:.4f} ({loss.avg:.4f}) \t '
                      'Training {metric_name} Score: {metric.val:.3f} ({metric.avg:.3f}) \t'
                      .format(j+1, len(train_data_loader), i + 1, batch_time=batch_time_t, data_time=data_time_t,
                              loss=losses_t, metric_name=metric_name, metric=metric_values_t))
            # Reset end timer
            end_time_t = time.time()
        
        # End of training epoch prints and updates
        print('Finished Training Epoch: {} \t '
              'Training Time: {batch_time.avg:.3f} \t '
              'Training Data Time: {data_time.avg:.3f}) \t '
              'Training Loss: {loss.avg:.4f} \t '
              'Training {metric_name} score: {metric.avg:.3f} \t'
              .format(i+1, batch_time=batch_time_t, data_time=data_time_t, loss=losses_t, metric_name=metric_name,
                      metric=metric_values_t))

        if cfg.logging.logging_enabled:
            log_train_metrics(experiment, losses_t.avg, metric_values_t.avg)

        # Validation

        # Only run validation every 10 epochs to save training time
        if i % 10 == 0:
            end_time_v = time.time()
            model.eval()
            if goal_type == 'regression':
                all_result_v = np.zeros((0))
            else:
                all_result_v = np.zeros((0,cfg.model.n_classes))
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
                    if goal_type == 'classification':
                        targets_v = targets_v.long().squeeze()
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
                    if goal_type == 'regression':
                        all_result_v = np.concatenate((all_result_v, outputs_v.squeeze(axis=1).cpu().detach().numpy()))
                        all_target_v = np.concatenate((all_target_v, targets_v.squeeze(axis=1).cpu().detach().numpy()))
                        all_loss_v = np.concatenate((all_loss_v, loss_v.cpu().squeeze(axis=1).detach().numpy()))
                    else:
                        all_result_v = np.concatenate((all_result_v, outputs_v.cpu().detach().numpy()))
                        all_target_v = np.concatenate((all_target_v, targets_v.cpu().detach().numpy()))
                        all_loss_v = np.concatenate((all_loss_v, loss_v.cpu().detach().numpy()))
                    all_uids_v = np.concatenate((all_uids_v, uids_v))

                else:
                    metric_targets_v = targets_v.cpu().detach().numpy()
                    metric_outputs_v = outputs_v.cpu().detach().numpy()
                    if goal_type == 'regression':
                        metric_v = r2_score(metric_targets_v, metric_outputs_v)
                    else:
                        predictions_v = np.argmax(metric_outputs_v, 1)
                        metric_v = accuracy_score(metric_targets_v, predictions_v)
                        top3_v = top_k_accuracy_score(metric_targets_t, metric_outputs_t, k=3)
                        top5_v = top_k_accuracy_score(metric_targets_t, metric_outputs_t, k=5)
                    metric_values_v.update(metric_v)
                    top3_values_v.update(top3_v)
                    top5_values_v.update(top5_v)
                    losses_v.update(loss_mean_v)

                    if k % 100 == 0:
                        print('Validation Batch: [{}/{}] in epoch: {} \t '
                              'Validation Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) \t '
                              'Validation Data Time: {data_time.val:.3f} ({data_time.avg:.3f}) \t '
                              'Validation Loss: {loss.val:.4f} ({loss.avg:.4f}) \t '
                              'Validation {metric_name} Top1: {metric.val:.3f} ({metric.avg:.3f}) Top3: {top3.val:.3f} ({top3.avg:.3f}) Top5: {top5.val:.3f} ({top5.avg:.3f})\t'
                              .format(k + 1, len(val_data_loader), i + 1, batch_time=batch_time_v, data_time=data_time_v,
                                      loss=losses_v, metric_name=metric_name, metric=metric_values_v, top3=top3_values_v, top5=top5_values_v))

                end_time_v = time.time()

            if cfg.evaluation.use_best_sample:
                # As results are over all possible combinations of views in each examination
                # each different combination needs to have a weight equal to its ratio.
                val_data = np.array((all_uids_v, all_loss_v))
                val_data = val_data.transpose(1, 0)
                pd_val_data = pd.DataFrame(val_data, columns=['us_id', 'loss'])
                pd_val_data['loss'] = pd_val_data['loss'].astype(np.float32)
                val_ue = pd_val_data.drop_duplicates(subset='us_id')[['us_id', 'loss']]
                all_mean_loss = []
                for ue in val_ue.itertuples():
                    exam_results = pd_val_data[pd_val_data['us_id'] == ue.us_id]
                    num_combinations = len(exam_results)
                    weight = 1/num_combinations
                    mean_exam_loss = exam_results['loss'].mean()
                    all_mean_loss.append(mean_exam_loss)
                    for indx in exam_results.index:
                        pd_val_data.loc[indx, 'metric_weight'] = weight
                np_loss = np.array(all_mean_loss, dtype=np.float32)
                loss_mean_v = np_loss.mean()
                weights = pd_val_data['metric_weight'].to_numpy()
                if goal_type == 'regression':
                    metric_v = r2_score(all_target_v, all_result_v, sample_weight=weights)
                else:
                    top3_v = top_k_accuracy_score(all_target_v.astype(np.int), all_result_v, k=3, sample_weight=weights)
                    top5_v = top_k_accuracy_score(all_target_v.astype(np.int), all_result_v, k=5, sample_weight=weights)
                    predictions_v = np.argmax(all_result_v, 1)
                    metric_v = accuracy_score(all_target_v.astype(np.int), predictions_v, sample_weight=weights)
            else:
                loss_mean_v = losses_v.avg
                metric_v = metric_values_v.avg

            # End of validation epoch prints and updates
            print('Finished Validation Epoch: {} \t '
                  'Validation Time: {batch_time.avg:.3f} \t '
                  'Validation Data Time: {data_time.avg:.3f} \t '
                  'Validation Loss: {loss:.4f} \t '
                  'Validation {metric_name} Top1: {metric:.3f} Top3: {top3:.3f} Top5: {top5:.3f}\t'
                  .format(i+1, batch_time=batch_time_v, data_time=data_time_v, loss=loss_mean_v, metric_name=metric_name
                          , metric=metric_v, top3=top3_v, top5=top5_v))
            if goal_type == 'regression':
                print('Example targets: {} \n Example outputs: {}'.format(torch.squeeze(targets_v), torch.squeeze(outputs_v)))
            
        if use_scheduler:
            scheduler.step(loss_mean_v)
            #scheduler.step()

        if cfg.training.checkpointing_enabled and cfg.logging.logging_enabled:
            if metric_v > max_val_metric:
                checkpoint_name = cfg.training.checkpoint_save_path + cfg.model.name + '_' + cfg.data.type + '_'\
                                  + cfg.data.name + '_exp_' + experiment.id + '.pth'
                save_checkpoint(checkpoint_name, model, optimizer)
                max_val_metric = metric_v
        elif cfg.training.checkpointing_enabled:
            if metric_v > max_val_metric:
                checkpoint_name = cfg.training.checkpoint_save_path + cfg.model.name + '_' + cfg.data.type + '_'\
                                  + cfg.data.name + '_test' + '.pth'
                save_checkpoint(checkpoint_name, model, optimizer)
                max_val_metric = metric_v

        if i % 10 == 0:
            if cfg.logging.logging_enabled:
                if goal_type == 'regression':
                    log_val_metrics(experiment, loss_mean_v, metric_v, max_val_metric)
                else:
                    log_val_classification(experiment, loss_mean_v, metric_v, max_val_metric, top3_v, top5_v)

        epoch_time = time.time() - start_time_epoch
        rem_epochs = cfg.training.epochs - (i+1)
        rem_time = rem_epochs * epoch_time
        print('Epoch {} completed. Time to complete: {}. Estimated remaining time: {}'.format(i+1, epoch_time, format_time(rem_time)))


def log_train_metrics(experiment, t_loss, t_metric):
    experiment.log_metric('training_loss', t_loss)
    experiment.log_metric('training_r2', t_metric)

def log_train_classification(experiment, t_loss, t_metric, top3, top5):
    experiment.log_metric('training_loss', t_loss)
    experiment.log_metric('training_top1_accuracy', t_metric)
    experiment.log_metric('training_top3_accuracy', top3)
    experiment.log_metric('training_top5_accuracy', top5)


def log_val_metrics(experiment, v_loss, v_metric, best_v_metric):
    experiment.log_metric('validation_loss', v_loss)
    experiment.log_metric('validation_r2', v_metric)
    experiment.log_metric('best_val_r2', best_v_metric)

def log_val_classification(experiment, loss, metric, max_val_metric, top3, top5):
    experiment.log_metric('validation_loss', loss)
    experiment.log_metric('validation_top1_accuracy', metric)
    experiment.log_metric('validation_top3_accuracy', top3)
    experiment.log_metric('validation_top5_accuracy', top5)
    experiment.log_metric('best_val_top1_accuracy', max_val_metric)

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

