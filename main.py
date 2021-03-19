import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from models import custom_cnn, resnext, i3d_bert, multi_stream
from training import train_and_validate, train
from validation import validate
import numpy as np
from hinge_loss import HingeLossRegression
import neptune
from data.npy_dataset import NPYDataset
from data.multi_stream_dataset import MultiStreamDataset
from arguments import get_args
import logging
import os

logging.basicConfig(level=logging.INFO)
logging.getLogger('numexpr').setLevel(logging.WARNING)

# Prevents deadlocks when using DDP and a DataLoader with multiple workers.
# Source https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html
mp.set_start_method('forkserver')


def main():

    args = get_args()
    
    assert args.model_name in ['ccnn', 'resnext', 'i3d', 'i3d_bert', 'i3d_bert_2stream']
    assert args.data_type in ['img', 'flow', 'multi-stream']

    setup_distributed_process(args)

    model, optimizer, tags = get_model_and_optimizer(args)

    train_data_set, val_data_set = create_data_sets(args)

    train_data_loader, val_data_loader = create_data_loaders(args, train_data_set, val_data_set)

    criterion = get_loss_function(args, train_data_set)

    experiment = None
    if args.logging.logging_enabled:
        neptune.init(args.logging.project_name)
        experiment_params = {**vars(args), 'train_dataset_size': len(train_data_set), 'val_dataset_size': len(val_data_set)}
        experiment = neptune.create_experiment(name=args.logging.experiment_name, params=experiment_params, tags=tags)

    if not os.path.exists(args.checkpoint_save_path):
        os.makedirs(args.checkpoint_save_path)

    for e in list(range(args.epochs)):
        phase = 'Training'
        loss_t, metric_t = train(args, model, optimizer, criterion, train_data_loader)
        log_metrics(experiment, loss_t, metric_t, phase)
        print_metrics(args, e, loss_t, metric_t, phase)
        if e % args.epochs_per_validation == 0:
            phase = 'Validation'
            loss_v, metric_v = validate(args, model, criterion, val_data_loader)
            log_metrics(experiment, loss_t, metric_t, phase)
            print_metrics(args, e, loss_t, metric_t, phase)

    cleanup_distributed_process()


def create_data_loaders(args, train_d_set, val_d_set):
    if args.data_loader.weighted_sampler:
        train_d_size = len(train_d_set)
        weights = [1.0] * train_d_size
        sampler = WeightedRandomSampler(weights=weights, num_samples=train_d_size, replacement=True)
    else:
        sampler = RandomSampler(train_d_set)
    train_data_loader = DataLoader(train_d_set, batch_size=args.data_loader.batch_size_train,
                                   num_workers=args.data_loader.n_workers, drop_last=args.data_loader.drop_last,
                                   sampler=sampler, pin_memory=True)

    val_data_loader = DataLoader(val_d_set, batch_size=args.data_loader.batch_size_eval,
                                 num_workers=args.data_loader.n_workers, drop_last=args.data_loader.drop_last, pin_memory=True)

    return train_data_loader, val_data_loader


def create_data_sets(args):
    if args.data.type == 'multi-stream':
        dataset_c = MultiStreamDataset
    else:
        dataset_c = NPYDataset
    # Create DataLoaders for training and validation
    train_d_set = dataset_c(args.data, args.transforms.train_t, args.augmentations.train_a, args.data.train_targets, is_eval_set=False)
    print("Training dataset size: {}".format(len(train_d_set)))

    val_d_set = dataset_c(args.data, args.transforms.eval_t, args.augmentations.eval_a, args.data.val_targets, is_eval_set=True)
    print("Validation dataset size: {}".format(len(val_d_set)))

    return train_d_set, val_d_set


def create_two_stream_models(args, checkpoint_img, checkpoint_flow):
    model_img = i3d_bert.rgb_I3D64f_bert2_FRMB(checkpoint_img, args.model.length,
                                               args.model.n_classes, args.model.n_input_channels_img,
                                               args.model.pre_n_classes, args.model.pre_n_input_channels_img)
    model_flow = i3d_bert.flow_I3D64f_bert2_FRMB(checkpoint_flow, args.model.length,
                                                 args.model.n_classes, args.model.n_input_channels_flow,
                                                 args.model.pre_n_classes, args.model.pre_n_input_channels_flow)
    return model_img, model_flow


def get_loss_function(args, data_set):
    if args.loss_function == 'hinge':
        criterion = HingeLossRegression(args.loss_epsilon, reduction=None)
    elif args.loss_function == 'mse':
        criterion = nn.MSELoss(reduction='none')
    elif args.loss_function == 'cross-entropy':
        # Get counts for each class
        # Instantiate class counts to 1 instead of 0 to prevent division by zero in case data is missing
        class_counts = np.array(args.model.n_classes*[1])
        for i in data_set.unique_exams['target'].value_counts().index:
            class_counts[i] = data_set.unique_exams['target'].value_counts().loc[i]
        # Calculate the inverse normalized ratio for each class
        weights = class_counts / class_counts.sum()
        weights = 1 / weights
        weights = weights / weights.sum()
        weights = torch.FloatTensor(weights).cuda()
        criterion = nn.CrossEntropyLoss(weight=weights, reduction='none')

    return criterion


def select_model_and_load_checkpoint(args):
    tags = []

    if args.model.name == 'ccnn':
        tags.append('CNN')
        model = custom_cnn.CNN()
    elif args.model.name == 'resnext':
        tags.append('ResNeXt')
        model = resnext.generate_model(model_depth=args.model.model_depth,
                                       cardinality=args.model.cardinality,
                                       n_classes=args.model.n_classes,
                                       n_input_channels=args.model.n_input_channels,
                                       shortcut_type=args.model.shortcut_type,
                                       conv1_t_size=args.model.conv1_t_size,
                                       conv1_t_stride=args.model.conv1_t_stride)
        model.load_state_dict(torch.load(args.model.pre_trained_checkpoint))
    elif args.model.name == 'i3d':
        tags.append('I3D')
        if args.training.continue_training:
            checkpoint = args.model.best_model
        else:
            checkpoint = args.model.pre_trained_checkpoint
        if args.data.type == 'img':
            tags.append('spatial')
            model = i3d_bert.inception_model(checkpoint, args.model.n_classes, args.model.n_input_channels,
                                             args.model.pre_n_classes, args.model.pre_n_input_channels)
        elif args.data.type == 'flow':
            tags.append('temporal')
            tags.append('TVL1')
            model = i3d_bert.inception_model_flow(checkpoint, args.model.n_classes, args.model.n_input_channels,
                                                  args.model.pre_n_classes, args.model.pre_n_input_channels)
    elif args.model.name == 'i3d_bert':
        tags.append('I3D')
        tags.append('BERT')
        if args.training.continue_training:
            state_dict = torch.load(args.model.best_model)['model']
            if args.data.type == 'img':
                tags.append('spatial')
                model = i3d_bert.rgb_I3D64f_bert2_FRMB('', args.model.length, args.model.n_classes,
                                                       args.model.n_input_channels, args.model.pre_n_classes,
                                                       args.model.pre_n_input_channels)
            if args.data.type == 'flow':
                tags.append('temporal')
                tags.append('TVL1')
                model = i3d_bert.flow_I3D64f_bert2_FRMB('', args.model.length, args.model.n_classes,
                                                        args.model.n_input_channels, args.model.pre_n_classes,
                                                        args.model.pre_n_input_channels)
            if args.data.type == 'multi-stream':
                tags.append('multi-stream')
                tags.append('TVL1')
                if args.model.shared_weights:
                    tags.append('shared-weights')
                    model_img, model_flow = create_two_stream_models(args, '', '')
                    model = multi_stream.MultiStreamShared(model_img, model_flow, len(state_dict['Linear_layer.weight'][0])/args.model.pre_n_classes, args.model.pre_n_classes)
                    model.load_state_dict(state_dict)
                    if not int(len(state_dict['Linear_layer.weight'][0])/args.model.pre_n_classes) == len(args.data.allowed_views) * 2 or not args.model.pre_n_classes == args.model.n_classes:
                        model.replace_fc(len(args.data.allowed_views) * 2, args.model.n_classes)
                        print('New FC shape:')
                        print(model._module['Linear_layer'].shape)
                else:
                    model_dict = {}
                    for view in args.data.allowed_views:
                        m_img_name = 'model_img_' + str(view)
                        m_flow_name = 'model_flow_' + str(view)
                        model_img, model_flow = create_two_stream_models(args, '', '')
                        model_dict[m_img_name] = model_img
                        model_dict[m_flow_name] = model_flow
                    model = multi_stream.MultiStream(model_dict)
        else:
            if args.data.type == 'img':
                tags.append('spatial')
                model = i3d_bert.rgb_I3D64f_bert2_FRMB(args.model.pre_trained_checkpoint, args.model.length,
                                                       args.model.n_classes, args.model.n_input_channels,
                                                       args.model.pre_n_classes, args.model.pre_n_input_channels)
            if args.data.type == 'flow':
                tags.append('temporal')
                tags.append('TVL1')
                model = i3d_bert.flow_I3D64f_bert2_FRMB(args.model.pre_trained_checkpoint, args.model.length,
                                                        args.model.n_classes, args.model.n_input_channels,
                                                        args.model.pre_n_classes, args.model.pre_n_input_channels)
            if args.data.type == 'multi-stream':
                tags.append('multi-stream')
                tags.append('TVL1')
                if args.model.shared_weights:
                    tags.append('shared-weights')
                    model_img, model_flow = create_two_stream_models(args, args.model.pre_trained_checkpoint_img,
                                                                     args.model.pre_trained_checkpoint_flow)
                    model = multi_stream.MultiStreamShared(model_img, model_flow, len(args.data.allowed_views)*2,
                                                           args.model.n_classes)
                else:
                    model_dict = {}
                    for view in args.data.allowed_views:
                        m_img_name = 'model_img_' + str(view)
                        m_flow_name = 'model_flow_' + str(view)
                        model_img, model_flow = create_two_stream_models(args, args.model.pre_trained_checkpoint_img,
                                                                         args.model.pre_trained_checkpoint_flow)
                        model_dict[m_img_name] = model_img
                        model_dict[m_flow_name] = model_flow
                    model = multi_stream.MultiStream(model_dict, args.model.n_classes)

    return model, tags


def get_model_and_optimizer(args):

    model, tags = select_model_and_load_checkpoint(args)

    model = optimize_model(args, model)

    # Wrap model in DDP
    model = DDP(model, device_ids=[args.local_rank])

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.optimizer.learning_rate, weight_decay=args.optimizer.weight_decay)

    return model, optimizer, tags


def optimize_model(args, model):
    model.to(args.local_rank)

    # CUDNN Auto-tuner. Use True when input size and model is static
    torch.backends.cudnn.benchmark = args.cuddn_auto_tuner

    if args.freeze_lower:
        for p in model.parameters():
            p.requires_grad = False
        model.Linear_layer.weight.requires_grad = True
        model.Linear_layer.bias.requires_grad = True
    return model


log_metrics(experiment, loss_t, metric_t, phase)
print_metrics(args, e, loss_t, metric_t, phase)


def setup_distributed_process(args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    torch.distributed.init_process_group('nccl')


def cleanup_distributed_process():
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
