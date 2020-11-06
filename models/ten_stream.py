from typing import Any

import torch.nn as nn
import torch
import torch.nn.functional as F


class TenStreamEnsemble(nn.Module):
    '''
    Ensemble for combining multiple models.
    If the models are already trained, all layers except the last fc_linear must be frozen.
    '''

    def __init__(self, model_2c_img, model_2c_flow, model_3c_img, model_3c_flow, model_4c_img, model_4c_flow,
                 model_lax_img, model_lax_flow, model_sax_img, model_sax_flow):
        super(TenStreamEnsemble, self).__init__()

        self.model_2c_img_name = 'Model_2c_img'
        self.model_2c_img = model_2c_img
        self.add_module(self.model_2c_img_name, self.model_2c_img)

        self.model_2c_flow_name = 'Model_2c_flow'
        self.model_2c_flow = model_2c_flow
        self.add_module(self.model_2c_flow_name, self.model_2c_flow)

        self.model_3c_img_name = 'Model_3c_img'
        self.model_3c_img = model_3c_img
        self.add_module(self.model_3c_img_name, self.model_3c_img)

        self.model_3c_flow_name = 'Model_3c_flow'
        self.model_3c_flow = model_3c_flow
        self.add_module(self.model_3c_flow_name, self.model_3c_flow)

        self.model_4c_img_name = 'Model_4c_img'
        self.model_4c_img = model_4c_img
        self.add_module(self.model_4c_img_name, self.model_4c_img)

        self.model_4c_flow_name = 'Model_4c_flow'
        self.model_4c_flow = model_4c_flow
        self.add_module(self.model_4c_flow_name, self.model_4c_flow)

        self.model_lax_img_name = 'Model_lax_img'
        self.model_lax_img = model_lax_img
        self.add_module(self.model_lax_img_name, self.model_lax_img)

        self.model_lax_flow_name = 'Model_lax_flow'
        self.model_lax_flow = model_lax_flow
        self.add_module(self.model_lax_flow_name, self.model_lax_flow)

        self.model_sax_img_name = 'Model_sax_img'
        self.model_sax_img = model_sax_img
        self.add_module(self.model_sax_img_name, self.model_sax_img)

        self.model_sax_flow_name = 'Model_sax_flow'
        self.model_sax_flow = model_sax_flow
        self.add_module(self.model_sax_flow_name, self.model_sax_flow)

        self.num_models = 10

        self.fc_name = 'Linear_layer'
        self.fc_linear = nn.Linear(self.num_models, 1)
        self.add_module(self.fc_name, self.fc_linear)

    def forward(self, input_tuple):
        assert len(input_tuple) == self.num_models

        x_2c_img = self.model_2c_img(input_tuple[0])
        x_2c_flow = self.model_2c_flow(input_tuple[1])

        x_3c_img = self.model_3c_img(input_tuple[2])
        x_3c_flow = self.model_3c_flow(input_tuple[3])

        x_4c_img = self.model_4c_img(input_tuple[4])
        x_4c_flow = self.model_4c_flow(input_tuple[5])

        x_lax_img = self.model_lax_img(input_tuple[6])
        x_lax_flow = self.model_lax_flow(input_tuple[7])

        x_sax_img = self.model_sax_img(input_tuple[8])
        x_sax_flow = self.model_sax_flow(input_tuple[9])


        x = torch.cat((x_2c_img, x_2c_flow, x_3c_img, x_3c_flow, x_4c_img, x_4c_flow, x_lax_img, x_lax_flow, x_sax_img, x_sax_flow), 1)

        x = self.fc_linear(F.relu(x))
        return x
