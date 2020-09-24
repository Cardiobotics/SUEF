from npy_dataset import NPYDataset
import torch
import hydra
from apex import amp
from apex.optimizers import FusedAdam
import os
from models import i3d
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:

    model_i3d_img_2c = i3d.InceptionI3d(cfg.model.n_classes, in_channels=1)
    model_i3d_flow_2c = i3d.InceptionI3d(cfg.model.n_classes, in_channels=2)

    opt_level = cfg.performance.amp_opt_level
    amp_enabled = cfg.performance.amp_enabled

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

    checkpoint_img = torch.load('path here')
    model_i3d_img_2c.load_state_dict(checkpoint_img['model'])

    for p in model_i3d_img_2c.parameters():
        p.requires_grad = False

    # Set optimizer
    if cuda_available and amp_enabled:
        optimizer_img = FusedAdam(filter(lambda p: p.requires_grad, model_i3d_img_2c.parameters()), lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
    else:
        optimizer_img = torch.optim.Adam(filter(lambda p: p.requires_grad, model_i3d_img_2c.parameters()), lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)

    optimizer_img.load_state_dict(checkpoint_img['optimizer'])

    model_i3d_img_2c, optimizer_img = amp.initialize(model_i3d_img_2c, optimizer_img, opt_level)
    amp.load_state_dict(checkpoint_img['amp'])








def save_checkpoint(save_file_path, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'amp': amp.state_dict()
    }
    torch.save(save_states, save_file_path)


def restore_checkpoint(args, model, optimizer, scheduler, checkpoint_path):
    # Restore
    checkpoint = torch.load(checkpoint_path)

    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, scheduler