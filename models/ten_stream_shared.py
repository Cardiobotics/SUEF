from typing import Any

import torch.nn as nn
import torch
import torch.nn.functional as F


class TenStreamShared(nn.Module):
    '''
    Ten-Stream model with shared weights.
    '''

    def __init__(self, model_img, model_flow, num_m):
        super(TenStreamShared, self).__init__()

        self.model_img_name = 'Model_img'
        self.model_img = model_img
        self.add_module(self.model_img_name, self.model_img)

        self.model_flow_name = 'Model_flow'
        self.model_flow = model_flow
        self.add_module(self.model_flow_name, self.model_flow)

        self.num_models = 8

        self.fc_name = 'Linear_layer'
        self.fc_linear = nn.Linear(self.num_models, 1)
        self.add_module(self.fc_name, self.fc_linear)

    def forward(self, x):
        assert len(x) == self.num_models

        y = torch.tensor()

        for i, input in enumerate(x):
            if i % 2 == 0:
                y = torch.cat((y, self.model_img(input)))
            else:
                y = torch.cat((y, self.model_flow(input)))

        y = self.fc_linear(F.relu(y))
        return y
