import numpy as np
from skimage.util import random_noise
from skimage.util import img_as_float32
from skimage.util import crop
from skimage.color import rgb2gray
from skimage.transform import resize


class DataAugmentations:
    def __init__(self, settings):
        super(DataAugmentations).__init__()
        
        self.settings = settings
        
    def transform(self, data):
        img = data.pixel_array

        # Some files had only this attribute instead of data.CineRate
        # so this was used instead, in theory they should be the same.
        original_fps = data.RecommendedDisplayFrameRate

        if self.settings['grayscale']:
            img = self.t_grayscale(img)
        if self.settings['float']:
            img = self.t_convert_to_float(img)
        if self.settings['noise']:
            img = self.t_gaussian_noise(img)
        if self.settings['speckle']:
            img = self.t_speckle(img)
        if self.settings['rescale_fps']:
            new_length = int(img.shape[0] * (self.settings['target_fps']/original_fps))
            img = self.t_resize(img, new_length, img.shape[1], img.shape[2])
        if self.settings['resize_frames']:
            img = self.t_resize(img, img.shape[0], self.settings['target_height'], self.settings['target_width'])
        if self.settings['crop']:
            img = self.t_crop(img, self.settings['crop'])

        return img

    def t_gaussian_noise(self, img):
        return random_noise(img, mode='gaussian', var=0.1)

    def t_grayscale(self, img):
        return np.expand_dims(rgb2gray(img), axis=3)

    def t_convert_to_float(self, img):
        return img_as_float32(img)

    def t_salt_and_pepper(self, img):
        return random_noise(img, mode='s&p')

    def t_speckle(self, img):
        return random_noise(img, mode='speckle', var=0.1)

    def t_resize(self, img, target_length, target_height, target_width):
        return resize(img, (target_length, target_height, target_width), mode='constant', cval=0, preserve_range=True,
                      anti_aliasing=True)

    def t_crop(self, img, crop_type):

        # Crop edges of frames
        if crop_type == 1:
            crop_sequence = ((0, 0), (int(img.shape[1] / 10), int(img.shape[1] / 10)),
                             (int(img.shape[2] / 20), int(img.shape[2] / 20)), (0, 0))

        # Crop length of video (last frames removed)
        if crop_type == 2 and img.shape[0] >= self.settings['target_length']:
            crop_sequence = ((0, img.shape[0] - self.settings['target_length']), (0, 0),
                             (0, 0), (0, 0))

        # Crop both edges and length
        if crop_type == 3:
            if img.shape[0] >= self.settings['target_length']:
                crop_sequence = ((0, img.shape[0] - self.settings['target_length']),
                                 (int(img.shape[1] / 10), int(img.shape[1] / 10)),
                                 (int(img.shape[2] / 20), int(img.shape[2] / 20)), (0, 0))
            else:
                crop_sequence = ((0, 0), (int(img.shape[1] / 10), int(img.shape[1] / 10)),
                                 (int(img.shape[2] / 20), int(img.shape[2] / 20)), (0, 0))

        return crop(img, crop_width=crop_sequence)
