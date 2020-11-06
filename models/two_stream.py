from typing import Any

import torch.nn as nn
import torch
import torch.nn.functional as F


class TwoStreamEnsemble(nn.Module):
    '''
    Ensemble for combining multiple models.
    If the models are already trained, all layers except the last fc_linear must be frozen.
    '''

    def __init__(self, model_img, model_flow):
        super(TwoStreamEnsemble, self).__init__()

        self.model_img_name = 'Model_img'
        self.model_img = model_img
        self.add_module(self.model_img_name, self.model_img)

        self.model_flow_name = 'Model_flow'
        self.model_flow = model_flow
        self.add_module(self.model_flow_name, self.model_flow)

        self.num_models = 2

        self.fc_name = 'Linear_layer'
        self.fc_linear = nn.Linear(self.num_models, 1)
        self.add_module(self.fc_name, self.fc_linear)

    def forward(self, input_tuple):
        assert len(input_tuple) == self.num_models

        x_img = self.model_img(input_tuple[0])
        x_flow = self.model_flow(input_tuple[1])

        x = torch.cat((x_img, x_flow), 1)

        print(x.shape)
        x = self.fc_linear(F.relu(x))
        return x
