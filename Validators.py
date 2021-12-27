import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import pandas as pd
import time
from sklearn.metrics import r2_score, accuracy_score, top_k_accuracy_score, mean_absolute_error, median_absolute_error, mean_squared_error
from utils.utils import AverageMeter, convert_EF_to_classes, convert_classes_to_EF
from utils.ddp_utils import is_master
from loss_functions.hinge_loss import HingeLossRegression
from loss_functions.ordinal_regression import get_ORAT_labels


class DDPValidator:
    def __init__(self, criterion, device, config, metric_name, goal_type):
        self.device = device
        self.criterion = criterion
        self.cfg = config
        self.metric_name = metric_name
        self.goal_type = goal_type

        self.use_half_prec = config.performance.half_precision

    @torch.no_grad()
    def validate(self, model, val_data_loader, curr_epoch):

        batch_time_v = AverageMeter()
        data_time_v = AverageMeter()
        losses_v = AverageMeter()
        metric_values_v = AverageMeter()
        if self.goal_type == 'classification':
            all_result_v = np.zeros((0, self.cfg.model.n_classes))
        elif self.goal_type == 'ordinal-regression':
            all_result_v = np.zeros((0))
        elif self.goal_type == 'regression':
            all_result_v = np.zeros((0))
            all_integer_outputs = np.zeros((0))
        all_target_v = np.zeros((0))
        all_uids_v = np.zeros((0))
        all_loss_v = np.zeros((0))
        end_time_v = time.time()
        model.eval()
        for k, (inputs_v, targets_v, _, uids_v, _) in enumerate(val_data_loader):
            # Update timer for data retrieval
            data_time_v.update(time.time() - end_time_v)

            # Move input to correct self.device
            if len(inputs_v) > 1:
                for p, inp in enumerate(inputs_v):
                    inputs_v[p] = inp.to(self.device, non_blocking=True)
            else:
                inputs_v = inputs_v.to(self.device, non_blocking=True)
            if self.goal_type == 'classification':
                targets_v = targets_v.long().squeeze()
            targets_v = targets_v.to(self.device, non_blocking=True)
            with torch.no_grad():
                # Get model validation output and validation loss
                with autocast(enabled=self.use_half_prec):
                    outputs_v = model(inputs_v)
                    if self.goal_type == 'ordinal-regression':
                        loss_v = self.criterion(outputs_v, targets_v, model.thresholds)
                    else:
                        loss_v = self.criterion(outputs_v, targets_v)
                    loss_mean_v = loss_v.mean()

            # Update timer for batch
            batch_time_v.update(time.time() - end_time_v)

            # Update metrics
            if self.cfg.evaluation.use_best_sample:
                if self.goal_type == 'classification':
                    all_result_v = np.concatenate((all_result_v, F.softmax(outputs_v, dim=1).cpu().detach().numpy()))
                    all_target_v = np.concatenate((all_target_v, targets_v.cpu().detach().numpy()))
                    all_loss_v = np.concatenate((all_loss_v, loss_v.cpu().detach().numpy()))
                elif self.goal_type == 'ordinal-regression':
                    all_result_v = np.concatenate((all_result_v, outputs_v.squeeze(axis=1).cpu().detach().numpy()))
                    all_target_v = np.concatenate((all_target_v, targets_v.squeeze(axis=1).cpu().detach().numpy()))
                    all_loss_v = np.concatenate((all_loss_v, loss_v.cpu().squeeze(axis=0).detach().numpy()))

                elif self.goal_type == 'regression':
                    all_result_v = np.concatenate((all_result_v, outputs_v.squeeze(axis=1).cpu().detach().numpy()))
                    all_target_v = np.concatenate((all_target_v, targets_v.squeeze(axis=1).cpu().detach().numpy()))
                    all_loss_v = np.concatenate((all_loss_v, loss_v.cpu().squeeze(axis=1).detach().numpy()))
                    # Convert model output to nearest integer of % 5 == 0
                    out_v = (np.around(outputs_v.squeeze(axis=1).cpu().detach().numpy()/5, decimals=0)*5).astype(np.int)
                    out_v[out_v < 20] = 20
                    out_v[out_v > 70] = 70
                    all_integer_outputs = np.concatenate((all_integer_outputs, out_v))
                all_uids_v = np.concatenate((all_uids_v, uids_v))

                if k % 100 == 0 and is_master():
                    print('Validation Batch: [{}/{}] in epoch: {} \t '
                          'Validation Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) \t '
                          'Validation Data Time: {data_time.val:.3f} ({data_time.avg:.3f}) \t '
                          .format(k + 1, len(val_data_loader), curr_epoch + 1, batch_time=batch_time_v, data_time=data_time_v))

            else:
                metric_targets_v = targets_v.cpu().detach().numpy()
                metric_outputs_v = outputs_v.cpu().detach().numpy()
                if self.goal_type == 'regression':
                    metric_v = r2_score(metric_targets_v, metric_outputs_v)
                elif self.goal_type == 'classification':
                    predictions_v = np.argmax(metric_outputs_v, 1)
                    metric_v = accuracy_score(metric_targets_v, predictions_v)
                elif self.goal_type == 'ordinal-regression':
                    labels_v = get_ORAT_labels(metric_outputs_v, model.thresholds)
                    metric_v = accuracy_score(metric_targets_v, labels_v)
                metric_values_v.update(metric_v)
                losses_v.update(loss_mean_v)

                if k % 100 == 0 and is_master():
                    print(('Validation Batch: [{}/{}] in epoch: {} \t '
                          'Validation Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) \t '
                          'Validation Data Time: {data_time.val:.3f} ({data_time.avg:.3f}) \t '
                          'Validation Loss: {loss.val:.4f} ({loss.avg:.4f}) \t '
                          'Validation' + self.metric_name +  ': {metric.val:.3f} ({metric.avg:.3f}) \t'
                          ).format(k + 1, len(val_data_loader), curr_epoch + 1, batch_time=batch_time_v, data_time=data_time_v,
                                  loss=losses_v, metric=metric_values_v))

            end_time_v = time.time()
        res = {}
        if self.cfg.evaluation.use_best_sample:
            # As results are over all possible combinations of views in each examination
            # each different combination needs to have a weight equal to its ratio.
            val_data = np.array((all_uids_v, all_target_v, all_loss_v))
            val_data = val_data.transpose(1, 0)
            pd_val_data = pd.DataFrame(val_data, columns=['us_id', 'target', 'loss'])
            pd_val_data[['target', 'loss']] = pd_val_data[['target', 'loss']].astype(np.float32)
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
            results = all_result_v
            weights = pd_val_data['metric_weight'].to_numpy()
            if self.goal_type == 'regression':
                metric_v = r2_score(targets, results, sample_weight=weights)
                metric_v_r2_integer = r2_score(targets, all_integer_outputs, sample_weight=weights)
                metric_v_mean_ae = mean_absolute_error(targets, all_integer_outputs, sample_weight=weights) 
                metric_v_median_ae = median_absolute_error(targets, all_integer_outputs, sample_weight=weights)
                val_mse = mean_squared_error(targets, all_integer_outputs, sample_weight=weights)
                target_classes = np.array([convert_EF_to_classes(t) for t in targets]).astype(np.int) 
                pred_classes = np.array([convert_EF_to_classes(p) for p in results]).astype(np.int)
                res['val/r2'] = metric_v
                res['val/accuracy'] = accuracy_score(target_classes, pred_classes, sample_weight=weights)
            elif self.goal_type == 'classification':
                predictions_v = np.argmax(results, 1)
                metric_v = accuracy_score(targets.astype(np.int), predictions_v, sample_weight=weights)
                preds_ef = np.array([convert_classes_to_EF(p) for p in predictions_v]).astype(np.int)
                targets_ef = np.array([convert_classes_to_EF(c) for c in targets]).astype(np.int)
                val_mse = mean_squared_error(targets_ef, preds_ef, sample_weight=weights) 
                metric_v_r2_integer = r2_score(targets_ef, preds_ef, sample_weight=weights)
                metric_v_mean_ae = mean_absolute_error(targets_ef, preds_ef, sample_weight=weights)
                metric_v_median_ae = median_absolute_error(targets_ef, preds_ef, sample_weight=weights)
                res['val/accuracy'] = metric_v
            elif self.goal_type == 'ordinal-regression':
                labels_v = get_ORAT_labels(results, model.thresholds)
                metric_v = accuracy_score(targets, labels_v)            
                preds_ef = np.array([convert_classes_to_EF(p) for p in labels_v]).astype(np.int)
                targets_ef = np.array([convert_classes_to_EF(c) for c in targets]).astype(np.int)
                val_mse = mean_squared_error(targets_ef, preds_ef, sample_weight=weights) 
                metric_v_r2_integer = r2_score(targets_ef, preds_ef, sample_weight=weights)
                metric_v_mean_ae = mean_absolute_error(targets_ef, preds_ef, sample_weight=weights)
                metric_v_median_ae = median_absolute_error(targets_ef, preds_ef, sample_weight=weights)
                res['val/accuracy'] = metric_v
        else:
            loss_mean_v = losses_v.avg
            metric_v = metric_values_v.avg
        
        res['val/mse_integer'] = val_mse
        res['val/r2_integer'] = metric_v_r2_integer
        res['val/mean_ae'] = metric_v_mean_ae
        res['val/median_ae'] = metric_v_median_ae
        res['val/loss'] = loss_mean_v
        # End of validation epoch prints and updates
        if is_master():
            print(('Finished Validation Epoch: {} \t '
                  'Validation Time: {batch_time.avg:.3f} \t '
                  'Validation Data Time: {data_time.avg:.3f} \t '
                  'Validation Loss: {loss:.4f} \t '
                  'Validation ' +  self.metric_name + ': {metric:.3f} \t'
                  ).format(curr_epoch + 1, batch_time=batch_time_v, data_time=data_time_v, loss=loss_mean_v, metric=metric_v))
            if self.goal_type == 'classification':
                outputs_v = torch.tensor(predictions_v)
            else:
                outputs_v = torch.squeeze(outputs_v)
            print('Example targets: {} \n Example outputs: {}'.format(torch.squeeze(targets_v), outputs_v))

        return res
