from npy_dataset import NPYDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import hydra
from utils import AverageMeter
import os
from models import i3d, ensemble
from omegaconf import DictConfig, OmegaConf
from npy_dataset import NPYDataset
import numpy as np
import pandas as pd
import neptune


@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:

    chkpt_path = '/home/ola/Projects/SUEF/saved_models/i3d_flow_3-Chamber.pth'
    data_path = '/media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/flow/2'
    target_path = '/home/ola/Projects/SUEF/data/train_inter_3c_2_4_flow.csv'
    result_path = '/home/ola/Projects/SUEF/results/train_inter_3c_2_4_flow.csv'

    model = i3d.InceptionI3d(cfg.model.n_classes, in_channels=2)

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

    checkpoint = torch.load(chkpt_path)
    model.load_state_dict(checkpoint['model'])

    if parallel_model:
        print("Available GPUS: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    for a in model.parameters():
        a.requires_grad = False

    model.eval()

    # Set loss criterion
    criterion = nn.MSELoss()

    test_data_set = NPYDataset(data_path, target_path, cfg.transforms, cfg.augmentations.eval, cfg.data.file_sep)

    test_data_loader = DataLoader(test_data_set, batch_size=cfg.data_loader.batch_size,
                                  num_workers=cfg.data_loader.n_workers, drop_last=cfg.data_loader.drop_last,
                                  shuffle=cfg.data_loader.shuffle)

    losses = AverageMeter()
    r2_scores = AverageMeter()

    saved_outputs = []
    saved_uids = []
    saved_targets = []

    for inputs, targets, uids in test_data_loader:

        # Move input to CUDA if available
        if cuda_available:
            targets = targets.to(device, non_blocking=True)
            inputs = inputs.to(device, non_blocking=True)

        with torch.no_grad():
            outputs = model(inputs)

        loss = criterion(outputs, targets)

        # Update metrics
        r2_targets = targets.cpu().detach()
        r2_outputs = outputs.cpu().detach()

        r2 = r2_score(r2_targets, r2_outputs)

        r2_scores.update(r2)

        losses.update(loss)

        saved_outputs.append(outputs.squeeze().cpu().numpy())
        saved_targets.append(targets.squeeze().cpu().numpy())
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

if __name__ == "__main__":
    main()
