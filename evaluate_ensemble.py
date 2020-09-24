from npy_dataset import NPYDataset
import torch
import hydra
from apex import amp
from apex.optimizers import FusedAdam
import os
from models import i3d, ensemble
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
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')

    model_i3d_img_2c.to(device)
    model_i3d_flow_2c.to(device)

    checkpoint_img = torch.load('/home/ola/Projects/SUEF/saved_models/i3d_img_2-Chamber.pth')
    model_i3d_img_2c.load_state_dict(checkpoint_img['model'])

    checkpoint_flow = torch.load('/home/ola/Projects/SUEF/saved_models/i3d_flow_2-Chamber.pth')
    model_i3d_flow_2c.load_state_dict(checkpoint_flow['model'])

    for a in model_i3d_img_2c.parameters():
        a.requires_grad = False
    for b in model_i3d_flow_2c.parameters():
        b.requires_grad = False

    ensemble_model = ensemble.FlexEnsemble(model_i3d_img_2c, model_i3d_flow_2c)
    ensemble_model.to(device)

    # Set optimizer
    if cuda_available and amp_enabled:
        optimizer = FusedAdam(filter(lambda p: p.requires_grad, ensemble_model.parameters()),
                              lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, ensemble_model.parameters()),
                                     lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)

    ensemble_model, optimizer = amp.initialize(ensemble_model, optimizer, opt_level=opt_level)





if __name__ == "__main__":
    main()
