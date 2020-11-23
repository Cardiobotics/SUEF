import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from models import custom_cnn, resnext, i3d, i3d_bert, two_stream, multi_stream
from training import train_and_validate
import neptune
import hydra
from data.npy_dataset import NPYDataset
from data.two_stream_dataset import TwoStreamDataset
from data.multi_stream_dataset import MultiStreamDataset
from omegaconf import DictConfig
import logging
import os

logging.basicConfig(level=logging.INFO)
logging.getLogger('numexpr').setLevel(logging.WARNING)



@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:

    assert cfg.model.name in ['ccnn', 'resnext', 'i3d', 'i3d_bert', 'i3d_bert_2stream']
    assert cfg.data.type in ['img', 'flow', 'two-stream', 'ten-stream', 'eight-stream', 'multi-stream']

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
        if cfg.data.type == 'img':
            tags.append('spatial')
            model = i3d.InceptionI3d(cfg.model.pre_n_classes, in_channels=cfg.model.n_input_channels)
            state_dict = torch.load(cfg.model.pre_trained_checkpoint)
            if not cfg.model.n_input_channels == cfg.model.pre_n_input_channels:
                conv1_weights = state_dict['Conv3d_1a_7x7.conv3d.weight']
                state_dict['Conv3d_1a_7x7.conv3d.weight'] = conv1_weights.mean(dim=1, keepdim=True)
            model.load_state_dict(state_dict)
            model.replace_logits(cfg.model.n_classes)
        elif cfg.data.type == 'flow':
            tags.append('temporal')
            tags.append('TVL1')
            model = i3d.InceptionI3d(cfg.model.pre_n_classes, in_channels=cfg.model.n_input_channels)
            state_dict = torch.load(cfg.model.pre_trained_checkpoint)
            model.load_state_dict(state_dict)
            model.replace_logits(cfg.model.n_classes)
    elif cfg.model.name == 'i3d_bert':
        tags.append('I3D')
        tags.append('BERT')
        if cfg.training.continue_training:
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
            state_dict = torch.load(cfg.model.best_model)['model']
            model.load_state_dict(state_dict)
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
            if cfg.data.type == 'two-stream':
                tags.append('2-stream')
                tags.append('TVL1')
                model = two_stream.TwoStreamEnsemble(create_two_stream_models(cfg, cfg.model.pre_trained_checkpoint_img,
                                                                              cfg.model.pre_trained_checkpoint_flow))
            if cfg.data.type == 'multi-stream':
                tags.append('multi-stream')
                tags.append('TVL1')
                if cfg.model.shared_weights:
                    tags.append('shared-weights')
                    model_img, model_flow = create_two_stream_models(cfg, cfg.model.pre_trained_checkpoint_img, cfg.model.pre_trained_checkpoint_flow)
                    model = multi_stream.MultiStreamShared(model_img, model_flow, len(cfg.data.allowed_views)*2)
                else:
                    model_dict = {}
                    for view in cfg.data.allowed_views:
                        m_img_name = 'model_img_' + str(view)
                        m_flow_name = 'model_flow_' + str(view)
                        model_img, model_flow = create_two_stream_models(cfg, cfg.model.pre_trained_checkpoint_img,
                                                                         cfg.model.pre_trained_checkpoint_flow)
                        model_dict[m_img_name] = model_img
                        model_dict[m_flow_name] = model_flow
                    model = multi_stream.MultiStream(model_dict)

    train_data_loader, val_data_loader = create_data_loaders(cfg)

    experiment = None
    if cfg.logging.logging_enabled:
        neptune.init(cfg.logging.project_name)
        experiment_params = {**dict(cfg.data_loader), **dict(cfg.transforms), **dict(cfg.augmentations),
                             **dict(cfg.performance), **dict(cfg.training), **dict(cfg.optimizer), **dict(cfg.model),
                             **dict(cfg.evaluation), 'data_stream': cfg.data.type, 'view': cfg.data.name,
                             'train_dataset_size': len(train_data_loader.dataset),
                             'val_dataset_size': len(val_data_loader.dataset)}
        experiment = neptune.create_experiment(name=cfg.logging.experiment_name, params=experiment_params, tags=tags)

    if not os.path.exists(cfg.training.checkpoint_save_path):
        os.makedirs(cfg.training.checkpoint_save_path)

    train_and_validate(model, train_data_loader, val_data_loader, cfg, experiment=experiment)


def create_data_loaders(cfg):
    if cfg.data.type == 'multi-stream':
        dataset_c = MultiStreamDataset
    else:
        dataset_c = NPYDataset
    # Create DataLoaders for training and validation
    train_d_set = dataset_c(cfg.data, cfg.transforms.train_t, cfg.augmentations.train_a, cfg.data.train_targets, is_eval_set=False)
    train_d_size = len(train_d_set)
    print("Training dataset size: {}".format(train_d_size))
    if cfg.data_loader.weighted_sampler:
        weights = [1.0] * train_d_size
        sampler = WeightedRandomSampler(weights=weights, num_samples=train_d_size, replacement=True)
    else:
        sampler = RandomSampler(train_d_set)
    train_data_loader = DataLoader(train_d_set, batch_size=cfg.data_loader.batch_size_train,
                                   num_workers=cfg.data_loader.n_workers, drop_last=cfg.data_loader.drop_last,
                                   sampler=sampler)

    val_d_set = dataset_c(cfg.data, cfg.transforms.eval_t, cfg.augmentations.eval_a, cfg.data.val_targets, is_eval_set=True)
    print("Validation dataset size: {}".format(len(val_d_set)))
    val_data_loader = DataLoader(val_d_set, batch_size=cfg.data_loader.batch_size_eval,
                                 num_workers=cfg.data_loader.n_workers, drop_last=cfg.data_loader.drop_last)

    return train_data_loader, val_data_loader


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
