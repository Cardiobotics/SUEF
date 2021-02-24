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

    target_path = cfg.data.train_targets
    result_path = '/home/ola/Projects/SUEF/results/8-stream-train-results.csv'

    state_dict = torch.load(cfg.model.best_model)['model']
    model_img, model_flow = create_two_stream_models(cfg, '', '')
    model = multi_stream.MultiStreamShared(model_img, model_flow, len(state_dict['Linear_layer.weight'][0]))
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

    for a in model.parameters():
        a.requires_grad = False

    model.eval()

    # Set loss criterion
    criterion = nn.MSELoss()

    val_d_set = MultiStreamDataset(cfg.data, cfg.transforms.eval_t, cfg.augmentations.eval_a, cfg.data.train_targets, is_eval_set=False)

    val_data_loader = DataLoader(val_d_set, batch_size=cfg.data_loader.batch_size_eval,
                                 num_workers=cfg.data_loader.n_workers, drop_last=cfg.data_loader.drop_last)

    losses = AverageMeter()
    r2_scores = AverageMeter()

    saved_outputs = []
    saved_uids = []
    saved_targets = []

    for inputs_v, targets_v, _, uids in val_data_loader:

        # Move input to CUDA if available
        if cuda_available:
            if len(inputs_v) > 1:
                for p, inp in enumerate(inputs_v):
                    inputs_v[p] = inp.to(device, non_blocking=True)
            else:
                inputs_v = inputs_v.to(device, non_blocking=True)
            targets_v = targets_v.to(device, non_blocking=True)

        with torch.no_grad():
            with autocast(enabled=use_half_prec):
                outputs_v = model(inputs_v)

        loss_v = criterion(outputs_v, targets_v)
        loss_mean_v = loss_v.mean()

        # Update metrics
        r2_targets = targets_v.cpu().detach()
        r2_outputs = outputs_v.cpu().detach()

        r2 = r2_score(r2_targets, r2_outputs)

        r2_scores.update(r2)

        losses.update(loss_mean_v)

        saved_outputs.append(outputs_v.squeeze().cpu().numpy())
        saved_targets.append(targets_v.squeeze().cpu().numpy())
        saved_uids.append(uids)

    saved_uids = np.array(saved_uids)
    saved_uids = saved_uids.reshape(saved_uids.size)
    saved_uids = np.expand_dims(saved_uids, axis=0)

    saved_outputs = np.array(saved_outputs)
    saved_outputs = saved_outputs.reshape(saved_outputs.size)
    saved_outputs = np.expand_dims(saved_outputs, axis=0)

    saved_targets = np.array(saved_targets)
    saved_targets = saved_targets.reshape(saved_targets.size)
    saved_targets = np.expand_dims(saved_targets, axis=0)

    uids_and_outputs = np.concatenate((saved_uids, saved_outputs, saved_targets))
    uids_and_outputs = uids_and_outputs.transpose(1, 0)
    pd_data = pd.DataFrame(uids_and_outputs, columns=['us_id', 'pred', 'target'])
    print(pd_data)
    pd_data.to_csv(result_path, sep=';', index=False)

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



