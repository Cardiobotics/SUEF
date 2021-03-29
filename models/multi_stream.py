from typing import Any

import torch.nn as nn
import torch
import torch.nn.functional as F


class MultiStream(nn.Module):
    '''
    Multi-Stream model.
    '''

    def __init__(self, model_dict, n_classes):
        super(MultiStream, self).__init__()

        self.endpoint_keys = []

        for model_name, model in model_dict.items():
            self.endpoint_keys.append(model_name)
            self.add_module(model_name, model)

        self.num_models = len(model_dict.items())
        self.n_classes = n_classes
        self.fc_input_size = self.num_models * self.n_classes

        self.fc_name = 'Linear_layer'
        fc_linear = nn.Linear(self.fc_input_size, n_classes)
        self.add_module(self.fc_name, fc_linear)

    def forward(self, x):
        assert len(x) == self.num_models

        y = []

        for inp, model_name in zip(x, self.endpoint_keys):
            y.append(self._modules[model_name](inp))
        y = torch.stack(y, dim=1).squeeze().view(-1, self.fc_input_size)
        y = self._modules[self.fc_name](F.relu(y))
        return y


class MultiStreamShared(nn.Module):
    '''
    Multi-Stream model with shared weights.
    '''

    def __init__(self, model_img, model_flow, num_m, n_classes):
        super(MultiStreamShared, self).__init__()

        self.model_img_name = 'Model_img'
        self.add_module(self.model_img_name, model_img)

        self.model_flow_name = 'Model_flow'
        self.add_module(self.model_flow_name, model_flow)

        self.num_models = num_m
        self.n_classes = n_classes
        self.fc_input_size = self.num_models * self.n_classes

        self.fc_name = 'Linear_layer'
        fc_linear = nn.Linear(self.fc_input_size, self.n_classes)
        self.add_module(self.fc_name, fc_linear)

    def forward(self, x):
        assert len(x) == self.num_models

        y = []

        for i, inp in enumerate(x):
            if i % 2 == 0:
                y.append(self._modules[self.model_img_name](inp))
            else:
                y.append(self._modules[self.model_flow_name](inp))
        y = torch.stack(y, dim=1).squeeze().view(-1, self.fc_input_size)
        y = self._modules[self.fc_name](F.relu(y))
        return y

    def replace_fc(self, num_models, n_classes):
        self.num_models = num_models
        self.n_classes = n_classes
        # Replace multistream fc layer
        self.fc_input_size = self.num_models * self.n_classes
        self.fc_name = 'Linear_layer'
        fc_linear = nn.Linear(self.fc_input_size, self.n_classes)
        self.add_module(self.fc_name, fc_linear)