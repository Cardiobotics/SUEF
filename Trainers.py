import torch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from sklearn.metrics import r2_score
import time
from utils.utils import AverageMeter
from utils.ddp_utils import is_master, is_dist_avail_and_initialized, AverageMeterDDP


class DDPTrainer:
    '''
    Trainer class for DDP trainer. Right now it does not to be a separate class but it makes it easier to extend it.
    '''
    def __init__(self, criterion, device, config):

        self.criterion = criterion
        self.device = device
        self.cfg = config

        # Mixed precision
        self.use_half_prec = config.performance.half_precision
        self.scaler = GradScaler(enabled=self.use_half_prec)

    def train_epoch(self, model, train_data_loader, optimizer, curr_epoch):

        batch_time_t = AverageMeterDDP()
        data_time_t = AverageMeterDDP()
        loss_values = AverageMeterDDP()
        r2_values = AverageMeterDDP()

        model.train()

        if is_dist_avail_and_initialized():
            train_data_loader.sampler.set_epoch(curr_epoch)

        end_time_t = time.time()
        for j, (inputs, targets, indexes, _) in enumerate(train_data_loader):
            # Update timer for data retrieval
            data_time_t.update([time.time() - end_time_t])
            # Move inputs and targets to correct device
            if len(inputs) > 1:
                for p, inp in enumerate(inputs):
                    if not torch.isfinite(inp).all():
                        raise ValueError('Input from dataloader not finite')
                    inputs[p] = inp.to(self.device, non_blocking=True)
            else:
                if not torch.isfinite(inputs).all():
                    raise ValueError('Input from dataloader not finite')
                inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Do forward and backwards pass
            # Get model train output and train loss
            with autocast(enabled=self.use_half_prec):
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss_mean = loss.mean()
            if self.cfg.data_loader.weighted_sampler:
                for index, l in zip(indexes, loss.cpu().detach()):
                    loss_ratio = l / loss_mean.cpu().detach()
                    loss_ratio = torch.clamp(loss_ratio, min=0.1, max=3)
                    train_data_loader.sampler.weights[index] = loss_ratio

            optimizer.zero_grad()

            # Backwards pass
            self.scaler.scale(loss_mean).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            # Calculate and update metrics
            try:
                if not torch.isfinite(outputs).all():
                    raise ValueError('Output from model not finite')
                metric_targets = targets.cpu().detach().numpy()
                metric_outputs = outputs.cpu().detach().numpy()
                metric_r2 = r2_score(metric_targets, metric_outputs)
                r2_values.update([metric_r2])
                loss_values.update([loss_mean])
            except ValueError as ve:
                print('Failed to calculate with error: {} and output: {}'.format(ve, outputs))

            # Update timer for batch
            batch_time_t.update([time.time() - end_time_t])

            if j % 100 == 0 and is_master():
                print('Training Batch: [{}/{}] in epoch: {} \t '
                      'Training Time: {batch_time.val[0]:.3f} ({batch_time.avg[0]:.3f}) \t '
                      'Training Data Time: {data_time.val[0]:.3f} ({data_time.avg[0]:.3f}) \t '
                      'Training Loss: {loss.val[0]:.4f} ({loss.avg[0]:.4f}) \t '
                      'Training R2 Score: {metric.val[0]:.3f} ({metric.avg[0]:.3f}) \t'
                      .format(j + 1, len(train_data_loader), curr_epoch + 1, batch_time=batch_time_t, data_time=data_time_t,
                              loss=loss_values, metric=r2_values))

            # Reset end timer
            end_time_t = time.time()

        if is_master():
            # End of training epoch prints and updates
            print('Finished Training Epoch: {} \t '
                  'Training Time: {batch_time.avg[0]:.3f} \t '
                  'Training Data Time: {data_time.avg[0]:.3f}) \t '
                  'Training Loss: {loss.avg[0]:.4f} \t '
                  'Training R2 score: {metric.avg[0]:.3f} \t'
                  .format(curr_epoch + 1, batch_time=batch_time_t, data_time=data_time_t, loss=loss_values,
                          metric=r2_values))

        return loss_values, r2_values