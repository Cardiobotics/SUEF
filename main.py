import argparse
from models import custom_cnn
from models import resnext
from models.training import train_and_validate
import config

def main():

    parser = argparse.ArgumentParser('Train a CNN model')
    parser.add_argument('--train_views', help='The file containing training file paths and their view labels')
    parser.add_argument('--train_targets', help='The path to the csv file containing targets for training data.')
    parser.add_argument('--train_target_sep', default=';', help='The separator for the train target csv file. Default is ;')
    parser.add_argument('--val_views', help='The file containing validation file paths and their view labels')
    parser.add_argument('--val_targets', help='The path to the csv file containing targets for validation data.')
    parser.add_argument('--val_target_sep', default=';', help='The separator for the val target csv file. Default is ;')
    parser.add_argument('--save_path', type=str, default='/home/ola/Projects/SUEF/models/resnext.pth', help='Path and filename to checkpoint the model as.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for the optimizer')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay for the optimizer')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for the DataLoader')
    parser.add_argument('--n_workers', type=int, default=11, help='Number of workers for the DataLoader')
    parser.add_argument('--model_type', type=str, default='resnext', help='Model type to train or evaluate. Possible options: ccnn, resnext')

    args = parser.parse_args()

    assert args.model_type in ['ccnn', 'resnext']

    if args.model_type == 'ccnn':
        model = custom_cnn.CNN()
    elif args.model_type == 'resnext':
        model = resnext.generate_model(model_depth=config.resnext_settings['model_depth'],
                                       cardinality=config.resnext_settings['cardinality'],
                                       n_classes=config.resnext_settings['n_classes'],
                                       n_input_channels=config.resnext_settings['n_input_channels'],
                                       shortcut_type=config.resnext_settings['shortcut_type'],
                                       conv1_t_size=config.resnext_settings['conv1_t_size'],
                                       conv1_t_stride=config.resnext_settings['conv1_t_stride'])

    train_and_validate(model, args)


if __name__ == "__main__":
    main()
