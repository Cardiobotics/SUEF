import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from models import custom_cnn, resnext, i3d, i3d_bert
from training import train_and_validate
import neptune
import hydra
from npy_dataset import NPYDataset
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:

    assert cfg.model.name in ['ccnn', 'resnext', 'i3d', 'i3d_bert']
    assert cfg.data.type in ['img', 'flow', '2stream']

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
        if cfg.data.type == 'img':
            model = i3d.InceptionI3d(cfg.model.pre_n_classes, in_channels=cfg.model.n_input_channels)
            state_dict = torch.load(cfg.model.pre_trained_checkpoint)
            if not cfg.model.n_input_channels == cfg.model.pre_n_input_channels:
                conv1_weights = state_dict['Conv3d_1a_7x7.conv3d.weight']
                state_dict['Conv3d_1a_7x7.conv3d.weight'] = conv1_weights.mean(dim=1, keepdim=True)
            model.load_state_dict(state_dict)
            model.replace_logits(cfg.model.n_classes)
        elif cfg.data.type == 'flow':
            model = i3d.InceptionI3d(cfg.model.pre_n_classes, in_channels=cfg.model.n_input_channels)
            state_dict = torch.load(cfg.model.pre_trained_checkpoint)
            model.load_state_dict(state_dict)
            model.replace_logits(cfg.model.n_classes)
    elif cfg.model.name == 'i3d_bert':
        if cfg.data.type == 'img':
            model = i3d_bert.rgb_I3D64f_bert2_FRMB(cfg.model.pre_trained_checkpoint, cfg.model.n_classes, cfg.model.length, cfg.model.n_input_channels)
        if cfg.data.type == 'flow':
            model = i3d_bert.flow_I3D64f_bert2_FRMB(cfg.model.pre_trained_checkpoint, cfg.model.n_classes, cfg.model.length)

    experiment = None
    if cfg.logging.logging_enabled:
        neptune.init(cfg.logging.project_name)
        experiment_params = {**dict(cfg.data_loader), **dict(cfg.transforms), **dict(cfg.augmentations),
                             **dict(cfg.performance), **dict(cfg.training), **dict(cfg.optimizer), **dict(cfg.model),
                             'data_stream': cfg.data.type, 'view': cfg.data.name}
        experiment = neptune.create_experiment(name=cfg.logging.experiment_name, params=experiment_params)

    train_data_loader, train_sampler, val_data_loader = create_data_loaders(cfg)

    train_and_validate(model, train_data_loader, train_sampler, val_data_loader, cfg, experiment=experiment)


def create_data_loaders(cfg):
    # Create DataLoaders for training and validation
    train_d_set = NPYDataset(cfg.data, cfg.transforms, cfg.augmentations.train, cfg.data.train_targets)
    train_d_size = len(train_d_set)
    print("Training dataset size: {}".format(train_d_size))
    if cfg.data_loader.weighted_sampler:
        weights = [1.0] * train_d_size
        sampler = WeightedRandomSampler(weights=weights, num_samples=train_d_size, replacement=False)
    else:
        sampler = RandomSampler(train_d_set)
    train_data_loader = DataLoader(train_d_set, batch_size=cfg.data_loader.batch_size,
                                   num_workers=cfg.data_loader.n_workers, drop_last=cfg.data_loader.drop_last,
                                   sampler=sampler)

    val_d_set = NPYDataset(cfg.data, cfg.transforms, cfg.augmentations.eval, cfg.data.val_targets)
    print("Validation dataset size: {}".format(len(val_d_set)))
    val_data_loader = DataLoader(val_d_set, batch_size=cfg.data_loader.batch_size,
                                 num_workers=cfg.data_loader.n_workers, drop_last=cfg.data_loader.drop_last)

    return train_data_loader, sampler, val_data_loader


if __name__ == "__main__":
    main()
