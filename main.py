import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from models import custom_cnn, resnext, i3d_bert, multi_stream
from training import train_and_validate
import neptune
from data.npy_dataset import NPYDataset
from data.multi_stream_dataset import MultiStreamDataset
import logging
import os
from arguments import get_args

logging.basicConfig(level=logging.INFO)
logging.getLogger('numexpr').setLevel(logging.WARNING)


def main():

    args = get_args()

    assert args.model_name in ['ccnn', 'resnext', 'i3d', 'i3d_bert']
    assert args.data_type in ['img', 'flow', 'multi-stream']

    tags = []

    if args.model_name == 'ccnn':
        tags.append('CNN')
        model = custom_cnn.CNN()
    elif args.model_name == 'i3d':
        tags.append('I3D')
        if args.continue_training:
            checkpoint = args.pre_trained_checkpoint
        else:
            checkpoint = ''
        if args.data_type == 'img':
            tags.append('spatial')
            model = i3d_bert.inception_model(checkpoint, args.n_outputs, args.n_input_channels_img)
        elif args.data_type == 'flow':
            tags.append('temporal')
            tags.append('TVL1')
            model = i3d_bert.inception_model_flow(checkpoint, args.n_outputs, args.n_input_channels_flow)
    elif args.model_name == 'i3d_bert':
        tags.append('I3D')
        tags.append('BERT')
        if args.continue_training:
            state_dict = torch.load(args.pre_trained_checkpoint)['model']
            if args.data_type == 'img':
                tags.append('spatial')
                model = i3d_bert.rgb_I3D64f_bert2_FRMB('', args.n_outputs, args.n_input_channels_img)
            if args.data_type == 'flow':
                tags.append('temporal')
                tags.append('TVL1')
                model = i3d_bert.flow_I3D64f_bert2_FRMB('', args.n_outputs, args.n_input_channels_flow)
            if args.data_type == 'multi-stream':
                tags.append('multi-stream')
                tags.append('TVL1')
                if args.shared_weights:
                    tags.append('shared-weights')
                    model_img, model_flow = create_two_stream_models(args, '', '')
                    model = multi_stream.MultiStreamShared(model_img, model_flow, len(state_dict['Linear_layer.weight'][0]))
                    model.load_state_dict(state_dict)
                    if not len(state_dict['Linear_layer.weight'][0]) == len(args.allowed_views) * 2:
                        model.replace_fc(len(args.allowed_views) * 2)
                else:
                    model_dict = {}
                    for view in args.allowed_views:
                        m_img_name = 'model_img_' + str(view)
                        m_flow_name = 'model_flow_' + str(view)
                        model_img, model_flow = create_two_stream_models(args, '', '')
                        model_dict[m_img_name] = model_img
                        model_dict[m_flow_name] = model_flow
                    model = multi_stream.MultiStream(model_dict)
        else:
            if args.data_type == 'img':
                tags.append('spatial')
                model = i3d_bert.rgb_I3D64f_bert2_FRMB(args.pre_trained_i3d_img, args.n_outputs,
                                                       args.n_input_channels_img)
            if args.data_type == 'flow':
                tags.append('temporal')
                tags.append('TVL1')
                model = i3d_bert.flow_I3D64f_bert2_FRMB(args.pre_trained_i3d_flow, args.n_outputs,
                                                        args.n_input_channels_flow)
            if args.data_type == 'multi-stream':
                tags.append('multi-stream')
                tags.append('TVL1')
                if args.shared_weights:
                    tags.append('shared-weights')
                    model_img, model_flow = create_two_stream_models(args, args.pre_trained_i3d_img,
                                                                     args.pre_trained_i3d_flow)
                    model = multi_stream.MultiStreamShared(model_img, model_flow, len(args.allowed_views)*2,
                                                           args.n_outputs)
                else:
                    model_dict = {}
                    for view in args.allowed_views:
                        m_img_name = 'model_img_' + str(view)
                        m_flow_name = 'model_flow_' + str(view)
                        model_img, model_flow = create_two_stream_models(args, args.pre_trained_i3d_img,
                                                                         args.pre_trained_i3d_flow)
                        model_dict[m_img_name] = model_img
                        model_dict[m_flow_name] = model_flow
                    model = multi_stream.MultiStream(model_dict, args.n_outputs)

    train_data_set, val_data_set = create_data_sets(args)

    train_data_loader, val_data_loader = create_data_loaders(args, train_data_set, val_data_set)

    experiment = None
    if args.logging_enabled:
        neptune.init(args.project_name)
        experiment_params = {**vars(args), 'train_dataset_size': len(train_data_loader.dataset),
                             'val_dataset_size': len(val_data_loader.dataset)}
        experiment = neptune.create_experiment(name=args.model_name, params=experiment_params, tags=tags)

    if not os.path.exists(args.checkpoint_save_path):
        os.makedirs(args.checkpoint_save_path)

    train_and_validate(model, train_data_loader, val_data_loader, args, experiment=experiment)


def create_data_loaders(args, train_d_set, val_d_set):
    if args.weighted_sampler:
        train_d_size = len(train_d_set)
        weights = [1.0] * train_d_size
        sampler = WeightedRandomSampler(weights=weights, num_samples=train_d_size, replacement=True)
    else:
        sampler = RandomSampler(train_d_set)
    train_data_loader = DataLoader(train_d_set, batch_size=args.train_batch_size, num_workers=args.n_workers,
                                   drop_last=args.drop_last, sampler=sampler, pin_memory=args.pin_memory)

    val_data_loader = DataLoader(val_d_set, batch_size=args.eval_batch_size, num_workers=args.n_workers,
                                 drop_last=args.drop_last, pin_memory=args.pin_memory)

    return train_data_loader, val_data_loader


def create_data_sets(args):
    if args.data_type == 'multi-stream':
        dataset_c = MultiStreamDataset
    else:
        dataset_c = NPYDataset
    # Create DataLoaders for training and validation
    train_d_set = dataset_c(args, is_eval_set=False)
    print("Training dataset size: {}".format(len(train_d_set)))

    val_d_set = dataset_c(args, is_eval_set=True)
    print("Validation dataset size: {}".format(len(val_d_set)))

    return train_d_set, val_d_set


def create_two_stream_models(args, checkpoint_img, checkpoint_flow):
    model_img = i3d_bert.rgb_I3D64f_bert2_FRMB(checkpoint_img, args.n_outputs, args.n_input_channels_img)
    model_flow = i3d_bert.flow_I3D64f_bert2_FRMB(checkpoint_flow, args.n_outputs, args.n_input_channels_flow,)
    return model_img, model_flow


if __name__ == "__main__":
    main()
