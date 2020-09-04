import numpy as np


class NPDataAugmentations:
    def __init__(self, flag_dict):
        super(NPDataAugmentations).__init__()

        self.flag_dict = flag_dict

    def transform(self, img):
        t_img = img
        if self.flag_dict['brightness']:
            t_img = self.t_brightness(t_img)

    def t_brighness(self, img):
        return img
