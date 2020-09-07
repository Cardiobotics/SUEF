import argparse
import cnn_model
import dcm_dataset
import torch
import torch.nn as nn
import re
from utils import AverageMeter, calculate_accuracy
import time
import config


def main():

    parser = argparse.ArgumentParser('Train a CNN model')
    parser.add_argument('--train_views', help='The file containing training file paths and their view labels')
    parser.add_argument('--train_targets', help='The path to the csv file containing targets for training data.')
    parser.add_argument('--train_target_sep', default=';', help='The separator for the train target csv file. Default is ;')
    parser.add_argument('--val_views', help='The file containing validation file paths and their view labels')
    parser.add_argument('--val_targets', help='The path to the csv file containing targets for validation data.')
    parser.add_argument('--val_target_sep', default=';', help='The separator for the val target csv file. Default is ;')
    parser.add_argument('--model_file', type=str, default='models/3dcnn.pth', help='Path and filename to checkpoint the model as.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay for the optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the DataLoader')
    parser.add_argument('--n_workers', type=int, default=8, help='Number of workers for the DataLoader')

    args = parser.parse_args()

    train_and_validate(args)


def train_and_validate(args):

    # Create DataLoaders for training and validation

    train_d_set = dcm_dataset.DCMDataset(args.train_views, args.train_targets, config.train_transforms,
                                         args.train_target_sep)
    train_data_loader = torch.utils.data.DataLoader(train_d_set, batch_size=args.batch_size, num_workers=args.n_workers)

    val_d_set = dcm_dataset.DCMDataset(args.val_views, args.val_targets, config.val_transforms,
                                         args.val_target_sep)
    val_data_loader = torch.utils.data.DataLoader(val_d_set, batch_size=args.batch_size, num_workers=args.n_workers)

    # Initialize model, loss function and optimizer

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    # CUDNN Auto-tuner. Use True when input size and model is static
    torch.backends.cudnn.benchmark = True


    model = cnn_model.Model_3DCNN()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.paramters(), lr=args.lr, weight_decay=args.wd)

    # Begin training

    for i in range(args.epochs):

        batch_time_t = AverageMeter()
        data_time_t = AverageMeter()
        losses_t = AverageMeter()
        accuracies_t = AverageMeter()

        batch_time_v = AverageMeter()
        data_time_v = AverageMeter()
        losses_v = AverageMeter()
        accuracies_v = AverageMeter()

        end_time_t = time.time()

        # Training
        model.train()
        for j, (inputs_t, targets_t) in enumerate(train_data_loader):
            # Update timer for data retrieval
            data_time_t.update(time.time() - end_time_t)
            
            # Move input to CUDA if available
            if cuda_available:
                targets_t = targets_t.to(device, non_blocking=True)
                inputs_t = inputs_t.to(device, non_blocking=True)

            # Get model train output and train loss
            outputs_t = model(inputs_t)
            loss_t = criterion(outputs_t, targets_t)
            
            # Update metrics
            acc_t = calculate_accuracy(outputs_t, targets_t)
            losses_t.update(loss_t)
            accuracies_t.update(acc_t)

            # Backwards pass and step
            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()

            # Update timer for batch
            batch_time_t.update(time.time(), end_time_t)
            end_time_t = time.time()
        
        # End of training epoch prints and updates
        print('Epoch: {} \t '
              'Training Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) \t '
              'Training Data Time {data_time.val:.3f} ({data_time.avg:.3f}) \t '
              'Training Loss {loss.val:.4f} ({loss.avg:.4f}) \t '
              'Training Acc {acc.val:.3f} ({acc.avg:.3f})'
              .format(i+1, batch_time=batch_time_t, data_time=data_time_t, loss=losses_t, acc=accuracies_t))
        end_time_v = time.time()
        
        
        # Validation
        model.eval()
        with torch.no_grad():
            for k, (inputs_v, targets_v) in enumerate(val_data_loader):
                # Update timer for data retrieval
                data_time_t.update(time.time() - end_time_v)

                # Move input to CUDA if available
                if cuda_available:
                    targets_v = targets_v.to(device, non_blocking=True)
                    inputs_v = inputs_v.to(device, non_blocking=True)

                # Get model validation output and validation loss
                outputs_v = model(inputs_v)
                loss_v = criterion(outputs_v, targets_v)

                # Update metrics
                acc_v = calculate_accuracy(outputs_v, targets_v)
                losses_v.update(loss_v)
                accuracies_v.update(acc_v)

                # Update timer for batch
                batch_time_v.update(time.time(), end_time_v)
                end_time_v = time.time()


        # End of validation epoch prints and updates
        print('Epoch: {} \t '
              'Validation Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) \t '
              'Validation Data Time {data_time.val:.3f} ({data_time.avg:.3f}) \t '
              'Validation Loss {loss.val:.4f} ({loss.avg:.4f}) \t '
              'Validation Acc {acc.val:.3f} ({acc.avg:.3f})'
              .format(i+1, batch_time=batch_time_v, data_time=data_time_v, loss=losses_v, acc=accuracies_v))
        end_time_t = time.time()