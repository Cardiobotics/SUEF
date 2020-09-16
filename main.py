import argparse
from models import custom_cnn
from models import resnext
from training import train_and_validate
import neptune
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:

    assert cfg.model.name in ['ccnn', 'resnext']

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
    if cfg.logging.logging_enabled:
        neptune.init(cfg.logging.project_name)
        experiment_params = {**dict(cfg.data_loader), **dict(cfg.transforms), **dict(cfg.augmentations),
                             **dict(cfg.performance), **dict(cfg.training), **dict(cfg.optimizer), **dict(cfg.model),
                             'view': cfg.data.name}
        neptune.create_experiment(name=cfg.logging.experiment_name, params=experiment_params)

    train_and_validate(model, cfg)


if __name__ == "__main__":
    main()
