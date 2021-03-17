import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from models import custom_cnn, resnext, i3d_bert, multi_stream
from training import train_and_validate
import neptune
import hydra
from data.npy_dataset import NPYDataset
from data.multi_stream_dataset import MultiStreamDataset
from omegaconf import DictConfig
import logging
import os

logging.basicConfig(level=logging.INFO)
logging.getLogger('numexpr').setLevel(logging.WARNING)


@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:

    assert cfg.model.name in ['ccnn', 'resnext', 'i3d', 'i3d_bert', 'i3d_bert_2stream']
    assert cfg.data.type in ['img', 'flow', 'multi-stream']

    tags = []

    if cfg.model.name == 'ccnn':
        tags.append('CNN')
        model = custom_cnn.CNN()
    elif cfg.model.name == 'resnext':
        tags.append('ResNeXt')
        model = resnext.generate_model(model_depth=cfg.model.model_depth,
                                       cardinality=cfg.model.cardinality,
                                       n_classes=cfg.model.n_classes,
                                       n_input_channels=cfg.model.n_input_channels,
                                       shortcut_type=cfg.model.shortcut_type,
                                       conv1_t_size=cfg.model.conv1_t_size,
                                       conv1_t_stride=cfg.model.conv1_t_stride)
        model.load_state_dict(torch.load(cfg.model.pre_trained_checkpoint))
    elif cfg.model.name == 'i3d':
        tags.append('I3D')
        if cfg.training.continue_training:
            checkpoint = cfg.model.best_model
        else:
            checkpoint = cfg.model.pre_trained_checkpoint
        if cfg.data.type == 'img':
            tags.append('spatial')
            model = i3d_bert.inception_model(checkpoint, cfg.model.n_classes, cfg.model.n_input_channels,
                                             cfg.model.pre_n_classes, cfg.model.pre_n_input_channels)
        elif cfg.data.type == 'flow':
            tags.append('temporal')
            tags.append('TVL1')
            model = i3d_bert.inception_model_flow(checkpoint, cfg.model.n_classes, cfg.model.n_input_channels,
                                                  cfg.model.pre_n_classes, cfg.model.pre_n_input_channels)
    elif cfg.model.name == 'i3d_bert':
        tags.append('I3D')
        tags.append('BERT')
        if cfg.training.continue_training:
            state_dict = torch.load(cfg.model.best_model)['model']
            if cfg.data.type == 'img':
                tags.append('spatial')
                model = i3d_bert.rgb_I3D64f_bert2_FRMB('', cfg.model.length, cfg.model.n_classes,
                                                       cfg.model.n_input_channels, cfg.model.pre_n_classes,
                                                       cfg.model.pre_n_input_channels)
            if cfg.data.type == 'flow':
                tags.append('temporal')
                tags.append('TVL1')
                model = i3d_bert.flow_I3D64f_bert2_FRMB('', cfg.model.length, cfg.model.n_classes,
                                                        cfg.model.n_input_channels, cfg.model.pre_n_classes,
                                                        cfg.model.pre_n_input_channels)
            if cfg.data.type == 'multi-stream':
                tags.append('multi-stream')
                tags.append('TVL1')
                if cfg.model.shared_weights:
                    tags.append('shared-weights')
                    model_img, model_flow = create_two_stream_models(cfg, '', '')
                    model = multi_stream.MultiStreamShared(model_img, model_flow, len(state_dict['Linear_layer.weight'][0])/cfg.model.pre_n_classes, cfg.model.pre_n_classes)
                    model.load_state_dict(state_dict)
                    if not int(len(state_dict['Linear_layer.weight'][0])/cfg.model.pre_n_classes) == len(cfg.data.allowed_views) * 2 or not cfg.model.pre_n_classes == cfg.model.n_classes:
                        model.replace_fc(len(cfg.data.allowed_views) * 2, cfg.model.n_classes)
                        print('New FC shape:')
                        print(model._module['Linear_layer'].shape)
                else:
                    model_dict = {}
                    for view in cfg.data.allowed_views:
                        m_img_name = 'model_img_' + str(view)
                        m_flow_name = 'model_flow_' + str(view)
                        model_img, model_flow = create_two_stream_models(cfg, '', '')
                        model_dict[m_img_name] = model_img
                        model_dict[m_flow_name] = model_flow
                    model = multi_stream.MultiStream(model_dict)
        else:
            if cfg.data.type == 'img':
                tags.append('spatial')
                model = i3d_bert.rgb_I3D64f_bert2_FRMB(cfg.model.pre_trained_checkpoint, cfg.model.length,
                                                       cfg.model.n_classes, cfg.model.n_input_channels,
                                                       cfg.model.pre_n_classes, cfg.model.pre_n_input_channels)
            if cfg.data.type == 'flow':
                tags.append('temporal')
                tags.append('TVL1')
                model = i3d_bert.flow_I3D64f_bert2_FRMB(cfg.model.pre_trained_checkpoint, cfg.model.length,
                                                        cfg.model.n_classes, cfg.model.n_input_channels,
                                                        cfg.model.pre_n_classes, cfg.model.pre_n_input_channels)
            if cfg.data.type == 'multi-stream':
                tags.append('multi-stream')
                tags.append('TVL1')
                if cfg.model.shared_weights:
                    tags.append('shared-weights')
                    model_img, model_flow = create_two_stream_models(cfg, cfg.model.pre_trained_checkpoint_img,
                                                                     cfg.model.pre_trained_checkpoint_flow)
                    model = multi_stream.MultiStreamShared(model_img, model_flow, len(cfg.data.allowed_views)*2,
                                                           cfg.model.n_classes)
                else:
                    model_dict = {}
                    for view in cfg.data.allowed_views:
                        m_img_name = 'model_img_' + str(view)
                        m_flow_name = 'model_flow_' + str(view)
                        model_img, model_flow = create_two_stream_models(cfg, cfg.model.pre_trained_checkpoint_img,
                                                                         cfg.model.pre_trained_checkpoint_flow)
                        model_dict[m_img_name] = model_img
                        model_dict[m_flow_name] = model_flow
                    model = multi_stream.MultiStream(model_dict, cfg.model.n_classes)

    train_data_set, val_data_set = create_data_sets(cfg)

    train_data_loader, val_data_loader = create_data_loaders(cfg, train_data_set, val_data_set)

    experiment = None
    if cfg.logging.logging_enabled:
        neptune.init(cfg.logging.project_name)
        experiment_params = {**dict(cfg.data_loader), **dict(cfg.transforms), **dict(cfg.augmentations),
                             **dict(cfg.performance), **dict(cfg.training), **dict(cfg.optimizer), **dict(cfg.model),
                             **dict(cfg.evaluation), 'target_file': cfg.data.train_targets, 'data_stream': cfg.data.type, 'view': cfg.data.name,
                             'train_dataset_size': len(train_data_loader.dataset),
                             'val_dataset_size': len(val_data_loader.dataset)}
        experiment = neptune.create_experiment(name=cfg.logging.experiment_name, params=experiment_params, tags=tags)

    if not os.path.exists(cfg.training.checkpoint_save_path):
        os.makedirs(cfg.training.checkpoint_save_path)

    train_and_validate(model, train_data_loader, val_data_loader, cfg, experiment=experiment)


def create_data_loaders(cfg, train_d_set, val_d_set):
    if cfg.data_loader.weighted_sampler:
        train_d_size = len(train_d_set)
        weights = [1.0] * train_d_size
        sampler = WeightedRandomSampler(weights=weights, num_samples=train_d_size, replacement=True)
    else:
        sampler = RandomSampler(train_d_set)
    train_data_loader = DataLoader(train_d_set, batch_size=cfg.data_loader.batch_size_train,
                                   num_workers=cfg.data_loader.n_workers, drop_last=cfg.data_loader.drop_last,
                                   sampler=sampler, pin_memory=True)

    val_data_loader = DataLoader(val_d_set, batch_size=cfg.data_loader.batch_size_eval,
                                 num_workers=cfg.data_loader.n_workers, drop_last=cfg.data_loader.drop_last, pin_memory=True)

    return train_data_loader, val_data_loader


def create_data_sets(cfg):
    if cfg.data.type == 'multi-stream':
        dataset_c = MultiStreamDataset
    else:
        dataset_c = NPYDataset
    # Create DataLoaders for training and validation
    train_d_set = dataset_c(cfg.data, cfg.transforms.train_t, cfg.augmentations.train_a, cfg.data.train_targets, is_eval_set=False)
    print("Training dataset size: {}".format(len(train_d_set)))

    val_d_set = dataset_c(cfg.data, cfg.transforms.eval_t, cfg.augmentations.eval_a, cfg.data.val_targets, is_eval_set=True)
    print("Validation dataset size: {}".format(len(val_d_set)))

    return train_d_set, val_d_set


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
