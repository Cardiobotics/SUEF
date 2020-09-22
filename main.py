import torch
from models import custom_cnn
from models import resnext
from models import i3d
from training import train_and_validate
import neptune
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:

    assert cfg.model.name in ['ccnn', 'resnext', 'i3d']
    assert cfg.data_stream.type in ['img', 'flow']

    if cfg.model.name == 'ccnn':
        model = custom_cnn.CNN()
    elif cfg.model.name == 'resnext':
        model = resnext.generate_model(model_depth=cfg.model.model_depth,
                                       cardinality=cfg.model.cardinality,
                                       n_classes=cfg.model.n_classes,
                                       n_input_channels=cfg.model.n_input_channels,
                                       shortcut_type=cfg.model.shortcut_type,
                                       conv1_t_size=cfg.model.conv1_t_size,
                                       conv1_t_stride=cfg.model.conv1_t_stride)
        model.load_state_dict(torch.load(cfg.model.pre_trained_checkpoint))
    elif cfg.model.name == 'i3d':
        if cfg.data_stream.type == 'img':
            model = i3d.InceptionI3d(cfg.model.pre_n_classes, in_channels=cfg.model.n_input_channels)
            state_dict = torch.load(cfg.model.pre_trained_checkpoint)
            conv1_weights = state_dict['Conv3d_1a_7x7.conv3d.weight']
            state_dict['Conv3d_1a_7x7.conv3d.weight'] = conv1_weights.mean(dim=1, keepdim=True)
            model.load_state_dict(state_dict)
            model.replace_logits(cfg.model.n_classes)
        elif cfg.data_stream.type == 'flow':
            model = i3d.InceptionI3d(cfg.model.pre_n_classes, in_channels=cfg.model.n_input_channels)
            state_dict = torch.load(cfg.model.pre_trained_checkpoint)
            model.load_state_dict(state_dict)
            model.replace_logits(cfg.model.n_classes)

    if cfg.logging.logging_enabled:
        neptune.init(cfg.logging.project_name)
        experiment_params = {**dict(cfg.data_loader), **dict(cfg.transforms), **dict(cfg.augmentations),
                             **dict(cfg.performance), **dict(cfg.training), **dict(cfg.optimizer), **dict(cfg.model),
                             'data_stream': cfg.data_stream.type, 'view': cfg.data.name}
        neptune.create_experiment(name=cfg.logging.experiment_name, params=experiment_params)

    train_and_validate(model, cfg)


if __name__ == "__main__":
    main()
