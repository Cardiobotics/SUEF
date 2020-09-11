import numpy as np
from skimage.util import random_noise
from skimage.util import img_as_float32
from skimage.util import crop
from skimage.color import rgb2gray
from skimage.transform import resize
import time


class DataAugmentations:
    def __init__(self, settings):
        super(DataAugmentations).__init__()
        
        self.settings = settings
        self.debug = False
        
    def transform(self, data):
        img = data.pixel_array
        time_start = time.time()
        if self.settings['grayscale']:
            img = self.t_grayscale(img)
        if self.settings['normalize_input']:
            img = self.t_normalize(img)
        if self.settings['noise']:
            img = self.t_gaussian_noise(img)
        if self.settings['speckle']:
            img = self.t_speckle(img)
        if self.debug:
            time_gf = time.time()
            time_gf_diff = time_gf - time_start
            print("Image size after grayscale: {}, Time to process: {}".format(img.shape, time_gf_diff))
        if self.settings['rescale_fps']:
            # Some files had only this attribute instead of data.CineRate
            # so this was used instead, in theory they should be the same.
            original_fps = data.RecommendedDisplayFrameRate
            if not self.settings['target_fps'] == original_fps:
                new_length = int(img.shape[0] * (self.settings['target_fps']/original_fps))
                img = self.t_resize(img, new_length, img.shape[1], img.shape[2])
        if self.debug:
            time_fps = time.time()
            time_fps_diff = time_fps - time_gf
            print("Image size after rescaling: {}, Time to process fps: {}".format(img.shape, time_fps_diff))
        if self.settings['resize_frames']:
            if not (img.shape[1] == self.settings['target_height'] and img.shape[2] == self.settings['target_width']):
                img = self.t_resize(img, img.shape[0], self.settings['target_height'], self.settings['target_width'])
        if self.debug:
            time_resize = time.time()
            time_resize_diff = time_resize - time_fps
            print("Image size after resizing: {}, Time to process: {}".format(img.shape, time_resize_diff))
        if self.settings['crop']:
            img = self.t_crop(img, self.settings['crop'])
        if self.debug:
            time_crop = time.time()
            time_crop_diff = time_crop - time_resize
            print("Image size after cropping: {}, Time to process: {}".format(img.shape, time_crop_diff))
        if self.settings['pad']:
            img = self.t_pad(img, self.settings['pad'])
        if self.debug:
            time_pad = time.time()
            time_pad_diff = time_pad - time_crop
            print("Image size after padding: {}, Time to process: {}".format(img.shape,time_pad_diff))
        return np.expand_dims(np.squeeze(img, axis=3), axis=0)

    def t_gaussian_noise(self, img):
        return random_noise(img, mode='gaussian', var=0.1)

    def t_grayscale(self, img):
        # Luminence numbers for converting RGB to grayscale
        b = [0.2989, 0.5870, 0.1140]
        img = np.dot(img[..., :3], b)
        return np.expand_dims(img, axis=3).astype(np.float32)

    def t_normalize(self, img):
        return img / 255

    def t_salt_and_pepper(self, img):
        return random_noise(img, mode='s&p')

    def t_speckle(self, img):
        return random_noise(img, mode='speckle', var=0.1)

    def t_resize(self, img, target_length, target_height, target_width):
        return resize(img, (target_length, target_height, target_width), mode='constant', cval=0, preserve_range=True,
                      anti_aliasing=True)

    def t_pad(self, img, pad_type):
        # Pad edges of frames
        if pad_type == 1 and (self.settings['target_height'] > img.shape[1] or self.settings['target_width'] > img.shape[2]):
            pad_sequence = ((0, 0),
                            (int((self.settings['target_height'] - img.shape[1])/2),
                            int((self.settings['target_height'] - img.shape[1])/2)),
                            (int((self.settings['target_width'] - img.shape[2])/2),
                             int((self.settings['target_width'] - img.shape[2])/2)),
                            (0, 0))
            img = np.pad(img, pad_width=pad_sequence)

        # Pad length of video (new frames added last)
        if pad_type == 2 and self.settings['target_length'] > img.shape[0]:
            pad_sequence = ((0, self.settings['target_length'] - img.shape[0]), (0, 0), (0, 0), (0, 0))
            img = np.pad(img, pad_width=pad_sequence)

        # Pad both edges and length
        if pad_type == 3 and (self.settings['target_length'] > img.shape[0] or
                              self.settings['target_height'] > img.shape[1] or
                              self.settings['target_width'] > img.shape[2]):

            pad_sequence = ((0, self.settings['target_length'] - img.shape[0]),
                            (int((self.settings['target_height'] - img.shape[1]) / 2),
                             int((self.settings['target_height'] - img.shape[1]) / 2)),
                            (int((self.settings['target_width'] - img.shape[2]) / 2),
                             int((self.settings['target_width'] - img.shape[2]) / 2)),
                            (0, 0))
            img = np.pad(img, pad_width=pad_sequence)

        return img

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
