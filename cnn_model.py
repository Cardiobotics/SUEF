import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Model_3DCNN(nn.Module):
    '''
    3D CNN model for regression.
    Takes input of shape: (batch_size, channels, length, height, width)
    '''

    def __init__(self, input_height, input_width, input_length, input_channels, batch_size):
        super(Model_3DCNN, self).__init__()

        num_filters_layer1 = 32
        num_filters_layer2 = 64
        num_filters_layer3 = 128
        num_filters_layer4 = 256
        num_filters_layer5 = 512

        kernel_size_layer1 = (7, 11, 11)
        kernel_size_layer2 = (5, 9, 9)
        kernel_size_layer3 = (5, 7, 7)
        kernel_size_layer4 = (3, 5, 5)
        kernel_size_layer5 = (3, 3, 3)

        filter_stride = 1
        filter_padding = 1

        pool_size = (3, 3, 3)
        pool_stride = 2
        pool_padding = 1

        self.lrelu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=pool_size, stride=pool_stride, padding=pool_padding)

        # Layer 1
        self.conv1 = nn.Conv3d(input_channels, num_filters_layer1, kernel_size=kernel_size_layer1, stride=filter_stride,
                               padding=filter_padding, bias=True)
        self.bn1 = nn.BatchNorm3d(num_filters_layer1)

        self.out_size_1_l = (input_length + 2 * filter_padding - (kernel_size_layer1[0] - 1) - 1) / filter_stride
        self.out_size_1_h = (input_height + 2 * filter_padding - (kernel_size_layer1[1] - 1) - 1) / filter_stride
        self.out_size_1_w = (input_width + 2 * filter_padding - (kernel_size_layer1[2] - 1) - 1) / filter_stride
        self.out_size_1 = (self.out_size_1_l, self.out_size_1_h, self.out_size_1_w)

        self.out_size_1_pooled_l = (self.out_size_1_l + 2 * pool_padding - (pool_size[0] - 1) - 1) / pool_stride + 1
        self.out_size_1_pooled_h = (self.out_size_1_h + 2 * pool_padding - (pool_size[1] - 1) - 1) / pool_stride + 1
        self.out_size_1_pooled_w = (self.out_size_1_w + 2 * pool_padding - (pool_size[2] - 1) - 1) / pool_stride + 1

        # Layer 2
        self.conv2 = nn.Conv3d(num_filters_layer1, num_filters_layer2, kernel_size=kernel_size_layer2, stride=filter_stride,
                               padding=filter_padding, bias=True)
        self.bn2 = nn.BatchNorm3d(num_filters_layer2)

        # Layer 3
        self.conv3 = nn.Conv3d(num_filters_layer2, num_filters_layer3, kernel_size=kernel_size_layer3, stride=filter_stride,
                               padding=filter_padding, bias=True)
        self.bn3 = nn.BatchNorm3d(num_filters_layer3)

        # Layer 4
        self.conv4 = nn.Conv3d(num_filters_layer3, num_filters_layer4, kernel_size=kernel_size_layer4, stride=filter_stride, padding=filter_padding, bias=True)
        self.bn4 = nn.BatchNorm3d(num_filters_layer4)

        # Layer 5
        self.conv5 = nn.Conv3d(num_filters_layer4, num_filters_layer5, kernel_size=kernel_size_layer5, stride=filter_stride, padding=filter_padding, bias=True)
        self.bn5 = nn.BatchNorm3d(num_filters_layer5)

        # Final classification layer
        self.fc = nn.Linear(6400, 1)

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 4
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x