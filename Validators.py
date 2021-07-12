import torch
from torch.cuda.amp import autocast
import numpy as np
import pandas as pd
import time
from sklearn.metrics import r2_score
from utils.utils import AverageMeter
from utils.ddp_utils import is_master


class DDPValidator:
    def __init__(self, criterion, device, config):
        self.device = device
        self.criterion = criterion
        self.cfg = config

        self.use_half_prec = config.performance.half_precision

    @torch.no_grad()
    def validate(self, model, val_data_loader, curr_epoch):

        model.eval()

        batch_time_v = AverageMeter()
        data_time_v = AverageMeter()
        losses_v = AverageMeter()
        metric_values_v = AverageMeter()

        all_result_v = np.zeros((0))
        all_target_v = np.zeros((0))
        all_uids_v = np.zeros((0))
        all_loss_v = np.zeros((0))

        end_time_v = time.time()
        
        for k, (inputs_v, targets_v, _, uids_v) in enumerate(val_data_loader):
            # Update timer for data retrieval
            data_time_v.update(time.time() - end_time_v)

            # Move input to correct self.device
            if len(inputs_v) > 1:
                for p, inp in enumerate(inputs_v):
                    inputs_v[p] = inp.to(self.device, non_blocking=True)
            else:
                inputs_v = inputs_v.to(self.device, non_blocking=True)
            targets_v = targets_v.to(self.device, non_blocking=True)

            with torch.no_grad():
                # Get model validation output and validation loss
                with autocast(enabled=self.use_half_prec):
                    outputs_v = model(inputs_v)
                    loss_v = self.criterion(outputs_v, targets_v)
                    loss_mean_v = loss_v.mean()

            # Update timer for batch
            batch_time_v.update(time.time() - end_time_v)

            # Update metrics
            if self.cfg.evaluation.use_best_sample:
                all_result_v = np.concatenate((all_result_v, outputs_v.squeeze(axis=1).cpu().detach().numpy()))
                all_target_v = np.concatenate((all_target_v, targets_v.squeeze(axis=1).cpu().detach().numpy()))
                all_loss_v = np.concatenate((all_loss_v, loss_v.cpu().squeeze(axis=1).detach().numpy()))
                all_uids_v = np.concatenate((all_uids_v, uids_v))

            else:
                metric_targets_v = targets_v.cpu().detach().numpy()
                metric_outputs_v = outputs_v.cpu().detach().numpy()
                metric_v = r2_score(metric_targets_v, metric_outputs_v)
                metric_values_v.update(metric_v)
                losses_v.update(loss_mean_v)

                if k % 100 == 0 and is_master():
                    print('Validation Batch: [{}/{}] in epoch: {} \t '
                          'Validation Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) \t '
                          'Validation Data Time: {data_time.val:.3f} ({data_time.avg:.3f}) \t '
                          'Validation Loss: {loss.val:.4f} ({loss.avg:.4f}) \t '
                          'Validation R2 Score: {metric.val:.3f} ({metric.avg:.3f}) \t'
                          .format(k + 1, len(val_data_loader), curr_epoch + 1, batch_time=batch_time_v, data_time=data_time_v,
                                  loss=losses_v, metric=metric_values_v))

            end_time_v = time.time()

        if self.cfg.evaluation.use_best_sample:
            # As results are over all possible combinations of views in each examination
            # each different combination needs to have a weight equal to its ratio.
            val_data = np.array((all_uids_v, all_result_v, all_target_v, all_loss_v))
            val_data = val_data.transpose(1, 0)
            pd_val_data = pd.DataFrame(val_data, columns=['us_id', 'result', 'target', 'loss'])
            pd_val_data[['result', 'target', 'loss']] = pd_val_data[['result', 'target', 'loss']].astype(np.float32)
            val_ue = pd_val_data.drop_duplicates(subset='us_id')[['us_id', 'target']]
            all_mean_loss = []
            for ue in val_ue.itertuples():
                exam_results = pd_val_data[pd_val_data['us_id'] == ue.us_id]
                num_combinations = len(exam_results)
                weight = 1 / num_combinations
                mean_exam_loss = exam_results['loss'].mean()
                all_mean_loss.append(mean_exam_loss)
                for indx in exam_results.index:
                    pd_val_data.loc[indx, 'metric_weight'] = weight
            np_loss = np.array(all_mean_loss, dtype=np.float32)
            loss_mean_v = np_loss.mean()
            targets = pd_val_data['target'].to_numpy()
            results = pd_val_data['result'].to_numpy()
            weights = pd_val_data['metric_weight'].to_numpy()
            metric_v = r2_score(targets, results, sample_weight=weights)
        else:
            loss_mean_v = losses_v.avg
            metric_v = metric_values_v.avg

        # End of validation epoch prints and updates
        if is_master():
            print('Finished Validation Epoch: {} \t '
                  'Validation Time: {batch_time.avg:.3f} \t '
                  'Validation Data Time: {data_time.avg:.3f} \t '
                  'Validation Loss: {loss:.4f} \t '
                  'Validation R2 score: {metric:.3f} \t'
                  .format(curr_epoch + 1, batch_time=batch_time_v, data_time=data_time_v, loss=loss_mean_v, metric=metric_v))
            print('Example targets: {} \n Example outputs: {}'.format(torch.squeeze(targets_v), torch.squeeze(outputs_v)))

        return loss_mean_v, metric_v
