import csv
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, DistributedSampler
from models import custom_cnn, resnext, i3d_bert, multi_stream
from data.npy_dataset import NPYDataset
from data.multi_stream_dataset import MultiStreamDataset, MultiStreamDatasetNoFlow
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from collections import OrderedDict
import utils.ddp_utils
import torch.nn as nn
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value

def load_filenames(path):
    files = []
    for dirName, _, fileList in os.walk(path):
        for filename in fileList:
            files.append((os.path.join(dirName, filename), filename))
    return files

def create_and_load_model_old(cfg):
    tags = []
    if cfg.performance.ddp:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.rank}
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
            state_dict = torch.load(cfg.model.best_model, map_location=map_location)['model']
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
                    model = multi_stream.MultiStreamShared(model_img, model_flow, len(state_dict['Linear_layer.weight'][0]), cfg.model.n_classes)
                    model.load_state_dict(state_dict)
                    if not len(state_dict['Linear_layer.weight'][0]) == len(cfg.data.allowed_views) * 2:
                        model.replace_fc(len(cfg.data.allowed_views) * 2)
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
                    
    return model, tags

def create_and_load_model(cfg):
    tags = []
    if cfg.performance.ddp:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.rank}
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
            if cfg.training.continue_training:
                checkpoint = cfg.model.best_model
            else:
                checkpoint = cfg.model.pre_trained_checkpoint
            model = i3d_bert.inception_model(checkpoint, cfg.model.n_classes, cfg.model.n_input_channels,
                                             cfg.model.pre_n_classes, cfg.model.pre_n_input_channels)
        elif cfg.data.type == 'flow':
            tags.append('temporal')
            tags.append('TVL1')
            if cfg.training.continue_training:
                checkpoint = cfg.model.best_model
            else:
                checkpoint = cfg.model.pre_trained_checkpoint
            model = i3d_bert.inception_model_flow(checkpoint, cfg.model.n_classes, cfg.model.n_input_channels,
                                                  cfg.model.pre_n_classes, cfg.model.pre_n_input_channels)
        elif cfg.data.type == 'multi-stream':
            tags.append('multi-stream')
            tags.append('TVL1')
            if cfg.model.shared_weights:
                tags.append('shared-weights')
                model_img = i3d_bert.Inception3D_Maxpool(cfg.model.pre_trained_checkpoint_img, cfg.model.n_classes,
                                                         cfg.model.n_input_channels_img, cfg.model.pre_n_classes,
                                                         cfg.model.pre_n_input_channels_img)
                model_flow = i3d_bert.Inception3D_Maxpool(cfg.model.pre_trained_checkpoint_flow, cfg.model.n_classes,
                                                         cfg.model.n_input_channels_flow, cfg.model.pre_n_classes,
                                                         cfg.model.pre_n_input_channels_flow)
                model = multi_stream.MultiStreamShared(model_img, model_flow, len(cfg.data.allowed_views) * 2,
                                                       cfg.model.n_classes)
                if cfg.optimizer.loss_function == 'all-threshold':
                    model.thresholds = torch.nn.Parameter(torch.tensor(range(10)).float(), requires_grad=True)
    elif cfg.model.name == 'i3d_bert':
        tags.append('I3D')
        tags.append('BERT')
        if cfg.training.continue_training:
            #state_dict = torch.load(cfg.model.best_model)['model']
            state_dict = torch.load(cfg.model.best_model, map_location=map_location)['model']
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
                    if cfg.optimizer.loss_function == 'all-threshold':
                        model.thresholds = torch.nn.Parameter(torch.tensor(range(10)).float(), requires_grad=True)
                else:
                    model_dict = {}
                    for view in cfg.data.allowed_views:
                        m_img_name = 'model_img_' + str(view)
                        m_flow_name = 'model_flow_' + str(view)
                        model_img, model_flow = create_two_stream_models(cfg, '', '')
                        model_dict[m_img_name] = model_img
                        model_dict[m_flow_name] = model_flow
                    model = multi_stream.MultiStream(model_dict)
            if cfg.data.type == 'no-flow':
                tags.append('no-flow')
                tags.append('multi-stream')
                if cfg.model.shared_weights:
                    tags.append('shared-weights')
                    model_img = i3d_bert.rgb_I3D64f_bert2_FRMB('', cfg.model.length,
                                                               cfg.model.n_classes, cfg.model.n_input_channels_img,
                                                               cfg.model.pre_n_classes,
                                                               cfg.model.pre_n_input_channels_img)
                    model = multi_stream.MSNoFlowShared(model_img, len(cfg.data.allowed_views)*2, cfg.model.pre_n_classes)
                    img_state_dict = OrderedDict({k: state_dict[k] for k in state_dict.keys() if
                                                  (k[0:9] == 'Model_img' or k[0:12] == 'Linear_layer')})
                    model.load_state_dict(img_state_dict)
                    model.replace_fc(len(cfg.data.allowed_views), cfg.model.n_classes)
                    if cfg.optimizer.loss_function == 'all-threshold':
                        model.thresholds = torch.nn.Parameter(torch.tensor(range(10)).float(), requires_grad=True)
                else:
                    model_dict = {}
                    for view in cfg.data.allowed_views:
                        m_img_name = 'model_img_' + str(view)
                        model_img = i3d_bert.rgb_I3D64f_bert2_FRMB('', cfg.model.length,
                                                                   cfg.model.n_classes, cfg.model.n_input_channels_img,
                                                                   cfg.model.pre_n_classes,
                                                                   cfg.model.pre_n_input_channels_img)
                        model_dict[m_img_name] = model_img
                    model = multi_stream.MultiStream(model_dict)
                    model.load_state_dict(state_dict)
            
    return model, tags

def create_criterion_and_optimizer(cfg, model, train_data_loader):
    # Create Criterion and Optimizer
    if cfg.optimizer.loss_function == 'hinge':
        metric_name = 'R2'
        goal_type = 'regression'
        # Set loss criterion
        criterion = HingeLossRegression(cfg.optimizer.loss_epsilon, reduction=None)
        # Hinge loss is dependent on L2 regularization so we cannot use AdamW
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.loss_function == 'mse':
        metric_name = 'R2'
        goal_type = 'regression'
        # Set loss criterion
        criterion = nn.MSELoss(reduction='none')
        # Set optimizer
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.loss_function == 'cross-entropy':
        metric_name = 'Accuracy'
        goal_type = 'classification'
        # Get counts for each class
        # Instantiate class counts to 1 instead of 0 to prevent division by zero in case data is missing
        class_counts = np.array(cfg.model.n_classes*[1])
        print(class_counts, cfg.model.n_classes, train_data_loader.dataset.unique_exams['target'].value_counts().index)
        for i in train_data_loader.dataset.unique_exams['target'].value_counts().index:
            class_counts[i] = train_data_loader.dataset.unique_exams['target'].value_counts().loc[i]
        # Calculate the inverse normalized ratio for each class
        weights = class_counts / class_counts.sum()
        weights = 1 / weights
        weights = weights / weights.sum()
        weights = torch.FloatTensor(weights).cuda()
        criterion = nn.CrossEntropyLoss(weight=weights, reduction='none')
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)

    elif cfg.optimizer.loss_function == 'all-threshold':
        metric_name = 'Accuracy'
        goal_type = 'ordinal-regression'
        # Get counts for each class
        # Instantiate class counts to 1 instead of 0 to prevent division by zero in case data is missing
        class_counts = np.array(len(train_data_loader.dataset.unique_exams['target'].unique())*[1])
        for i in train_data_loader.dataset.unique_exams['target'].value_counts().index:
            class_counts[i] = train_data_loader.dataset.unique_exams['target'].value_counts().loc[i]
        # Calculate the inverse normalized ratio for each class
        weights = class_counts / class_counts.sum()
        weights = 1 / weights
        weights = weights / weights.sum()
        weights = torch.FloatTensor(weights).cuda()
        criterion = OrdinalRegressionAT(sample_weights=weights, reduction=None)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)

    return criterion, optimizer, goal_type

def create_data_loaders(cfg, train_d_set, val_d_set):
    train_data_loader = create_train_loader(cfg, train_d_set)
    val_data_loader = create_val_loader(cfg, val_d_set)

    return train_data_loader, val_data_loader


def create_train_loader(cfg, train_d_set):
    if cfg.performance.ddp:
        if cfg.data_loader.weighted_sampler:
            t_sampler = utils.ddp_utils.DistributedWeightedSampler(train_d_set)
        else:
            t_sampler = DistributedSampler(train_d_set)
    elif cfg.data_loader.weighted_sampler:
        train_d_size = len(train_d_set)
        weights = [1.0] * train_d_size
        t_sampler = WeightedRandomSampler(weights=weights, num_samples=train_d_size, replacement=True)
    else:
        t_sampler = RandomSampler(train_d_set)
    train_data_loader = DataLoader(train_d_set, batch_size=cfg.data_loader.batch_size_train,
                                   num_workers=cfg.data_loader.n_workers, drop_last=cfg.data_loader.drop_last,
                                   sampler=t_sampler,
                                   pin_memory=True)
    return train_data_loader


def create_val_loader(cfg, val_d_set):
    val_data_loader = DataLoader(val_d_set, batch_size=cfg.data_loader.batch_size_eval,
                                 num_workers=cfg.data_loader.n_workers, drop_last=cfg.data_loader.drop_last)
    return val_data_loader

def create_data_sets(cfg):
    if cfg.data.type == 'multi-stream':
        dataset_c = MultiStreamDataset
    elif cfg.data.type == 'no-flow':
        dataset_c = MultiStreamDatasetNoFlow
    else:
        dataset_c = NPYDataset
    # Create DataLoaders for training and validation
    train_d_set = dataset_c(cfg.data, cfg.transforms.train_t, cfg.augmentations.train_a, cfg.data.train_targets, is_eval_set=False)
    print("Training dataset size: {}".format(len(train_d_set)))

    val_d_set = dataset_c(cfg.data, cfg.transforms.eval_t, cfg.augmentations.eval_a, cfg.data.val_targets, is_eval_set=True)
    print("Validation dataset size: {}".format(len(val_d_set)))

    return train_d_set, val_d_set

def create_data_sets_old(cfg):
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

def update_cfg(cfg, key, val):
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        setattr(cfg, key, val)

def log_train_metrics(experiment, t_loss, t_metric, lr):
    experiment['train/loss'].log(t_loss)
    experiment['train/r2'].log(t_metric)
    experiment['train/lr'].log(lr)


def log_train_classification(experiment, t_loss, t_metric, top3, top5):
    experiment['train/loss'].log(t_loss)
    experiment['train/top1_accuracy'].log(t_metric)
    experiment['train/top3_accuracy'].log(top3)
    experiment['train/top5_accuracy'].log(top5)


def log_val_metrics_old(experiment, v_loss, v_metric, best_v_metric):
    experiment['val/loss'].log(v_loss)
    experiment['val/r2'].log(v_metric)
    experiment['val/best_r2'].log(best_v_metric)

def log_val_metrics(experiment, res):
    for k, v in res.items():
        experiment[k].log(v)

def log_val_classification(experiment, loss, metric, max_val_metric, top3, top5):
    experiment['val/loss'].log(loss)
    experiment['val/top1_accuracy'].log(metric)
    experiment['val/top3_accuracy'].log(top3)
    experiment['val/top5_accuracy'].log(top5)
    experiment['val/best_top1_accuracy'].log(max_val_metric)

def update_val_results(results, **kwargs):
    for k, v in kwargs.items():
        results[k] = v
    
def save_checkpoint(save_file_path, model, optimizer):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(save_states, save_file_path)


def save_checkpoint(save_file_path, model, optimizer):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(save_states, save_file_path)
