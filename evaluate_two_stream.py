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
import neptune


@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:

    if cfg.logging.logging_enabled:
        neptune.init(cfg.logging.project_name)
        experiment_params = {**dict(cfg.data_loader), **dict(cfg.transforms), **dict(cfg.augmentations),
                             **dict(cfg.performance), **dict(cfg.training), **dict(cfg.optimizer), **dict(cfg.model),
                             'data_stream': cfg.data_stream.type, 'view': cfg.data.name}
        neptune.create_experiment(name=cfg.logging.experiment_name, params=experiment_params)

    model_i3d_img_2c = i3d.InceptionI3d(cfg.model.n_classes, in_channels=cfg.model.n_input_channels_img)
    model_i3d_flow_2c = i3d.InceptionI3d(cfg.model.n_classes, in_channels=cfg.model.n_input_channels_flow)

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

    model_i3d_img_2c.to(device)
    model_i3d_flow_2c.to(device)

    checkpoint_img = torch.load(cfg.model.pre_trained_checkpoint_img)
    model_i3d_img_2c.load_state_dict(checkpoint_img['model'])

    checkpoint_flow = torch.load(cfg.model.pre_trained_checkpoint_flow)
    model_i3d_flow_2c.load_state_dict(checkpoint_flow['model'])

    if parallel_model:
        print("Available GPUS: {}".format(torch.cuda.device_count()))
        model_i3d_img_2c = nn.DataParallel(model_i3d_img_2c)
        model_i3d_flow_2c = nn.DataParallel(model_i3d_flow_2c)

    for a in model_i3d_img_2c.parameters():
        a.requires_grad = False
    for b in model_i3d_flow_2c.parameters():
        b.requires_grad = False

    model_i3d_img_2c.eval()
    model_i3d_flow_2c.eval()

    # Set loss criterion
    criterion = nn.MSELoss()

    val_data_img = NPYDataset('/media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/img/20',
                              '/home/ola/Projects/SUEF/data/val_targets_img_20.csv', cfg.transforms,
                              cfg.augmentations.eval, cfg.data.file_sep)
    val_data_flow = NPYDataset('/media/ola/324ac400-018f-4d6b-941e-361b54f3e5f6/flow/20',
                               '/home/ola/Projects/SUEF/data/val_targets_flow_20.csv', cfg.transforms,
                               cfg.augmentations.eval, cfg.data.file_sep)

    val_data_loader_img = DataLoader(val_data_img, batch_size=cfg.data_loader.batch_size,
                                     num_workers=cfg.data_loader.n_workers, drop_last=cfg.data_loader.drop_last,
                                     shuffle=cfg.data_loader.shuffle)
    val_data_loader_flow = DataLoader(val_data_flow, batch_size=cfg.data_loader.batch_size,
                                      num_workers=cfg.data_loader.n_workers, drop_last=cfg.data_loader.drop_last,
                                      shuffle=cfg.data_loader.shuffle)

    img_losses = AverageMeter()
    flow_losses = AverageMeter()
    avg_losses = AverageMeter()
    img_r2 = AverageMeter()
    flow_r2 = AverageMeter()
    avg_r2 = AverageMeter()

    for (img_inputs, img_targets), (flow_inputs, flow_targets) in zip(val_data_loader_img, val_data_loader_flow):
        assert torch.eq(img_targets, flow_targets).all()

        targets = img_targets

        # Cast input to half precision
        targets = targets.half()
        img_inputs = img_inputs.half()
        flow_inputs = flow_inputs.half()

        # Move input to CUDA if available
        if cuda_available:
            targets = targets.to(device, non_blocking=True)
            img_inputs = img_inputs.to(device, non_blocking=True)
            flow_inputs = flow_inputs.to(device, non_blocking=True)

        with torch.no_grad():
            flow_outputs = model_i3d_flow_2c(flow_inputs)
            img_outputs = model_i3d_img_2c(img_inputs)

        avg_outputs = (flow_outputs + img_outputs)/2

        img_loss = criterion(img_outputs, targets)
        flow_loss = criterion(flow_outputs, targets)
        avg_loss = criterion(avg_outputs, targets)

        # Update metrics
        r2_targets = targets.cpu().detach()
        r2_img_outputs = img_outputs.cpu().detach()
        r2_flow_outputs = flow_outputs.cpu().detach()
        r2_avg_outputs = avg_outputs.cpu().detach()
        r2_img = r2_score(r2_targets, r2_img_outputs)
        r2_flow = r2_score(r2_targets, r2_flow_outputs)
        r2_avg = r2_score(r2_targets, r2_avg_outputs)
        img_r2.update(r2_img)
        flow_r2.update(r2_flow)
        avg_r2.update(r2_avg)

        img_losses.update(img_loss)
        flow_losses.update(flow_loss)
        avg_losses.update(avg_loss)

    print("Img r2: {}".format(img_r2.avg))
    print("Flow r2: {}".format(flow_r2.avg))
    print("Avg r2: {}".format(avg_r2.avg))

    print("Img loss: {}".format(img_losses.avg))
    print("Flow loss: {}".format(flow_losses.avg))
    print("Avg loss: {}".format(avg_losses.avg))

    if cfg.logging.logging_enabled:
        neptune.log_metric('loss', avg_losses.avg)
        neptune.log_metric('r2', avg_r2.avg)


if __name__ == "__main__":
    main()

