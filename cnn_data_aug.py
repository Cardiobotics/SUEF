import numpy as np


class DataAugmentations:
    def __init__(self, flag_dict):
        super(DataAugmentations).__init__()

        self.flag_dict = flag_dict

    def transform(self, img):
        t_img = img
        if self.flag_dict['gray_scale']:
            t_img = self.t_rgb2gray(t_img)
        if self.flag_dict['brightness']:
            t_img = self.t_brightness(t_img)

    def t_brighness(self, img):
        return img

    def t_rgb2gray(img):
        '''
        Converts a 3-channel RGB image to grayscale.
        :param img: image data in array form (np.arr or python arr)
        :return: The greyscale img in numpy.array form.
        '''
        # Luminence numbers for converting RGB to grayscale
        b = [0.2989, 0.5870, 0.1140]
        gray_img = np.dot(img[..., :3], b)
        gray_img = np.expand_dims(gray_img, axis=0)
        return gray_img.astype(np.uint8)

