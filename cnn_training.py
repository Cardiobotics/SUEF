import argparse
import cnn_model
import npy_dataset
import torch
import torch.nn as nn
import re


def main():

    parser = argparse.ArgumentParser('Train a CNN model')
    parser.add_argument('--train_data', help='The folder containing training data')
    parser.add_argument('--train_targets', help='The path to the csv file containing targets for training data.')
    parser.add_argument('--train_target_sep', default=';', help='The separator for the train target csv file. Default is ;')
    parser.add_argument('--val_data', help='The folder containing validation data')
    parser.add_argument('--val_targets', help='The path to the csv file containing targets for validation data.')
    parser.add_argument('--val_target_sep', default=';', help='The separator for the val target csv file. Default is ;')
    parser.add_argument('--model_file', help='Path and filename to checkpoint the model as.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay for the optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the DataLoader')
    parser.add_argument('--n_workers', type=int, default=8, help='Number of workers for the DataLoader')
    parser.add_argument('--img_size', type=str, default='300,300,300', help='Image size in H*W*L. Accepted formats: '
                                                                            'H*W*L, HxWxL, HXWXL, H,W,L. '
                                                                            'Example: 350x450x20')

    args = parser.parse_args()

    re_img_size = re.search('([0-9]+)[xX*,]([0-9]+)[xX*,]([0-9]+)', args.img_size)

    img_h = re_img_size[0]
    img_w = re_img_size[1]
    img_l = re_img_size[2]

    img_size = (img_h, img_w, img_l)

    transform_flags = {'brightness': True, 'greyscale': True}

    train_and_validate(args, transform_flags)


def train_and_validate(args, transform_flags, img_size):

    if transform_flags['greyscale']:
        img_channels = 1
    else:
        img_channels = 3

    # Create DataLoaders for training and validation

    train_data_loader = generate_data_loader(args.train_data, args.batch_size, args.n_workers, args.train_targets,
                                             transform_flags, args.train_target_sep)
    val_data_loader = generate_data_loader(args.val_data, args.batch_size, args.n_workers, args.val_targets,
                                           transform_flags, args.val_target_sep)

    # Initialize model, loss function and optimizer

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    model = cnn_model.Model_3DCNN(img_size[0], img_size[1], img_size[2], img_channels, args.batch_size)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.paramters(), lr=args.lr, weight_decay=args.wd)

    # Begin training

    for i in range(args.epochs):

        # Training
        model.train()
        for j, (t_inputs, t_targets) in enumerate(train_data_loader):

            # Move input to CUDA if available
            if cuda_available:
                t_targets = t_targets.to(device, non_blocking=True)
                t_inputs = t_inputs.to(device, non_blocking=True)

            # Get model train output and train loss
            t_outputs = model(t_inputs)
            t_loss = criterion(t_outputs, t_targets)

            # Backwards pass and step
            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()



        # Validation
        model.eval()
        with torch.no_grad():
            for k, (v_inputs, v_targets) in enumerate(val_data_loader):

                # Move input to CUDA if available
                if cuda_available:
                    v_targets = v_targets.to(device, non_blocking=True)
                    v_inputs = v_inputs.to(device, non_blocking=True)

                # Get model validation output and validation loss
                v_outputs = model(v_inputs)
                v_loss = criterion(v_outputs, v_targets)





def generate_data_loader(data_path, batch_size, n_workers, target_file, t_flags, target_file_sep=';', uid_len=13):
    '''
    Returns a DataLoader object created from a CustomDataset using the supplied arguments.
    :param data_path: The folder of the npy files containing image data (Str)
    :param batch_size: Batch size for the dataloader (Int)
    :param n_workers: Number of workers for the dataloader (Int)
    :param target_file: The target csv file containing user ids and corresponding targets (Str)
    :param t_flags: Transform flags for the dataset (Dict)
    :param target_file_sep: The separator for the csv file (Str)
    :param uid_len: The length of the userid that prefixes the npy files (Int)
    :return: DataLoader object
    '''
    d_set = npy_dataset.NPYDataset(data_path, target_file, target_file_sep, uid_len, t_flags)
    d_loader = torch.utils.data.DataLoader(d_set, batch_size=batch_size, num_workers=n_workers)

    return d_loader