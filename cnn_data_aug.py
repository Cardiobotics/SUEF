import numpy as np
from skimage.util import random_noise
from skimage.util import img_as_float32
from skimage.color import rgb2gray

class DataAugmentations:
    def __init__(self, flag_dict):
        super(DataAugmentations).__init__()

        self.flag_dict = flag_dict

    def transform(self, img):
        if self.flag_dict['grayscale']:
            img = self.t_grayscale(img)
        if self.flag_dict['float']:
            img = self.t_convert_to_float(img)
        if self.flag_dict['noise']:
            img = self.t_gaussian_noise(img)
        if self.flag_dict['speckle']:
            img = self.t_speckle(img)
        return img

    def t_gaussian_noise(self, img):
        return random_noise(img, mode='gaussian', var=0.1)

    def t_grayscale(self, img):
        return rgb2gray(img)

    def t_convert_to_float(self, img):
        return img_as_float32(img)

    def t_salt_and_pepper(self, img):
        return random_noise(img, mode='s&p')

    def t_speckle(self, img):
        return random_noise(img, mode='speckle', var=0.1)