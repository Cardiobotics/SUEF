import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import r2_score
import hydra
from utils.utils import AverageMeter
import os
from models import i3d_bert, multi_stream
from omegaconf import DictConfig
from data.multi_stream_dataset import MultiStreamDataset
import numpy as np
import pandas as pd


@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:

    result_folder = '/home/ola/Projects/SUEF/results/'
    result_name = cfg.model.best_model[-47:-4] + '_test_results.csv'
    result_path = os.path.join(result_folder, result_name)

    state_dict = torch.load(cfg.model.best_model)['model']
    model_img, model_flow = create_two_stream_models(cfg, '', '')
    model = multi_stream.MultiStreamShared(model_img, model_flow, len(state_dict['Linear_layer.weight'][0]), cfg.model.n_classes)
    model.load_state_dict(state_dict)

    use_half_prec = cfg.performance.half_precision

    # Set visible devices
    parallel_model = cfg.performance.parallel_mode
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Set cuda
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        #torch.cuda.empty_cache()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)

    if parallel_model:
        print("Available GPUS: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model.eval()

    # Set loss criterion
    criterion = nn.MSELoss(reduction='none')

    test_d_set = MultiStreamDataset(cfg.data, cfg.transforms.eval_t, cfg.augmentations.eval_a, cfg.data.test_targets, is_eval_set=True)

    test_data_loader = DataLoader(test_d_set, batch_size=cfg.data_loader.batch_size_eval,
                                 num_workers=cfg.data_loader.n_workers, drop_last=False)

    model.eval()

    all_result_t = np.zeros((0,))
    all_target_t = np.zeros((0,))
    all_uids_t = np.zeros((0,))
    all_loss_t = np.zeros((0,))
    all_2c_iids_t = np.zeros((0,))
    all_3c_iids_t = np.zeros((0,))
    all_4c_iids_t = np.zeros((0,))
    all_lax_iids_t = np.zeros((0,))

    for k, (inputs_t, targets_t, _, uids_t, iids_t) in enumerate(test_data_loader):

        # Move input to correct self.device
        if len(inputs_t) > 1:
            for p, inp in enumerate(inputs_t):
                inputs_t[p] = inp.to(device, non_blocking=True)
        else:
            inputs_t = inputs_t.to(device, non_blocking=True)
        targets_t = targets_t.to(device, non_blocking=True)

        with torch.no_grad():
            # Get model validation output and validation loss
            with autocast(enabled=use_half_prec):
                outputs_t = model(inputs_t)
                loss_t = criterion(outputs_t, targets_t)

        # Update metrics
        # We gather all information for that specific combination during evaluation and calculate metrics at the end

        all_result_t = np.concatenate((all_result_t, outputs_t.squeeze(axis=1).cpu().detach().numpy()))
        all_target_t = np.concatenate((all_target_t, targets_t.squeeze(axis=1).cpu().detach().numpy()))
        all_loss_t = np.concatenate((all_loss_t, loss_t.cpu().squeeze(axis=1).detach().numpy()))
        all_uids_t = np.concatenate((all_uids_t, uids_t))
        # iids is a list like this [2c_iid, 3c_iid, 4c_iid, lax_iid] where iids can be either numeric or None
        # we save these to keep track on the specific combination of views that was evaluated.
        np_iids = torch.stack(iids_t).numpy()
        all_2c_iids_t = np.concatenate((all_2c_iids_t, np_iids[0, :]))
        all_3c_iids_t = np.concatenate((all_3c_iids_t, np_iids[1, :]))
        all_4c_iids_t = np.concatenate((all_4c_iids_t, np_iids[2, :]))
        all_lax_iids_t = np.concatenate((all_lax_iids_t, np_iids[3, :]))

        if k % 100 == 0:
            print('Test Batch: [{}/{}] completed \t '
                  .format(k + 1, len(test_data_loader)))

    # As results are over all possible combinations of views in each examination
    # each different combination needs to have a weight equal to its ratio.
    val_data = np.array((all_uids_t, all_2c_iids_t, all_3c_iids_t, all_4c_iids_t, all_lax_iids_t, all_result_t, all_target_t, all_loss_t), dtype=object)
    val_data = val_data.transpose(1, 0)
    pd_val_data = pd.DataFrame(val_data, columns=['us_id', '2c_iid', '3c_iid', '4c_iid', 'lax_iid', 'result', 'target', 'loss'])
    pd_val_data[['result', 'target', 'loss']] = pd_val_data[['result', 'target', 'loss']].astype(np.float32)
    val_ue = pd_val_data.drop_duplicates(subset='us_id')[['us_id', 'target']]
    all_mean_loss = []
    # For every unique examination (us_id), we calculate a weight based on the number of view combinations
    # that examination has. This weight is used to calculate r2.
    for ue in val_ue.itertuples():
        exam_results = pd_val_data[pd_val_data['us_id'] == ue.us_id]
        num_combinations = len(exam_results)
        weight = 1 / num_combinations
        mean_exam_loss = exam_results['loss'].mean()
        all_mean_loss.append(mean_exam_loss)
        for indx in exam_results.index:
            pd_val_data.loc[indx, 'r2_weight'] = weight
    # Loss is calculated as mean over the mean of each examination as otherwise
    # results from examinations with many combinations would overshadow results from examinations with only one
    np_loss = np.array(all_mean_loss, dtype=np.float32)
    loss_mean_t = np_loss.mean()
    targets = pd_val_data['target'].to_numpy()
    results = pd_val_data['result'].to_numpy()
    weights = pd_val_data['r2_weight'].to_numpy()
    metric_t = r2_score(targets, results, sample_weight=weights)

    pd_val_data.to_csv(result_path, sep=';', index=False)
    print('Evaluation of test data complete. \t'
          'Mean MSE loss: {} \t'
          'Weighted R2 score: {}'
          .format(loss_mean_t, metric_t))


def create_two_stream_models(cfg, checkpoint_img, checkpoint_flow):
    model_img = i3d_bert.rgb_I3D64f_bert2_FRMB(checkpoint_img, cfg.model.length,
                                               cfg.model.n_classes, cfg.model.n_input_channels_img,
                                               cfg.model.pre_n_classes, cfg.model.pre_n_input_channels_img)
    model_flow = i3d_bert.flow_I3D64f_bert2_FRMB(checkpoint_flow, cfg.model.length,
                                                 cfg.model.n_classes, cfg.model.n_input_channels_flow,
                                                 cfg.model.pre_n_classes, cfg.model.pre_n_input_channels_flow)
    return model_img, model_flow

if __name__ == "__main__":
    main()



