import numpy as np
from skimage.util import random_noise
from skimage.util import img_as_float32
from skimage.util import crop
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.transform import rotate
from random import choice
from random import randint
import time
from omegaconf import OmegaConf, DictConfig
import cv2


class DataAugmentations:
    def __init__(self, transforms, augmentations):
        super(DataAugmentations).__init__()
        
        self.transforms = transforms
        self.augmentations = augmentations
        self.debug = False

    def transform_values(self, img):

        # Pixel values expected to be in range 0-255
        if self.transforms.normalize_input:
            img = self.t_normalize_signed(img)

        # Add some kind of noise to the image
        if self.augmentations.gaussian_noise:
            img = self.t_gaussian_noise(img)
        if self.augmentations.speckle:
            img = self.t_speckle(img)
        if self.augmentations.salt_and_pepper:
            img = self.t_salt_and_pepper(img)

        # Shift the image in some way
        if self.augmentations.translate_h:
            img = self.t_translate_h(img)
        if self.augmentations.translate_v:
            img = self.t_translate_v(img)
        if self.augmentations.rotate:
            img = self.t_rotate(img)

        return img.astype(np.float32)

    def transform_size(self, img, fps, hr):
        time_start = time.time()
        if self.transforms.grayscale:
            img = self.t_grayscale_mean(img)
        if self.debug:
            time_gf = time.time()
            time_gf_diff = time_gf - time_start
            print("Image size after grayscale: {}, Time to process: {}".format(img.shape, time_gf_diff))
        if self.transforms.rescale_fps or self.transforms.resize_frames or self.transforms.rescale_fphb:
            assert not (self.transforms.rescale_fps and self.transforms.rescale_fphb)

            # Rescale length by either fps or fphb
            if self.transforms.rescale_fps and not self.transforms.target_fps == fps:
                new_length = int(img.shape[0] * (self.transforms.target_fps/fps))
            elif self.transforms.rescale_fphb:
                curr_fphb = self.calc_fphb(hr, fps)
                new_length = int(img.shape[0]*(self.transforms.target_fphb/curr_fphb))
            else:
                new_length = img.shape[0]

            # Rescale size
            if self.transforms.resize_frames and not (img.shape[1] == self.transforms.target_height and
                                                      img.shape[2] == self.transforms.target_width):
                new_height = self.transforms.target_height
                new_width = self.transforms.target_width
            else:
                new_height = img.shape[1]
                new_width = img.shape[2]

            img = self.t_resize(img, new_length, new_height, new_width)

        if self.debug:
            time_rescale = time.time()
            time_fps_diff = time_rescale - time_gf
            print("Image size after rescaling: {}, Time to process: {}".format(img.shape, time_fps_diff))
        if self.transforms.crop_sides or self.transforms.crop_length:
            img = self.t_crop(img)
        if self.debug:
            time_crop = time.time()
            time_crop_diff = time_crop - time_rescale
            print("Image size after cropping: {}, Time to process: {}".format(img.shape, time_crop_diff))
        if self.transforms.loop_length:
            img = self.t_loop_length(img)
        if self.debug:
            time_loop = time.time()
            time_loop_diff = time_loop - time_crop
            print("Image size after length looping: {}, Time to process: {}".format(img.shape, time_loop_diff))
        return img

    def t_grayscale_custom(self, img):
        # Luminence numbers for converting RGB to grayscale
        b = [0.2989, 0.5870, 0.1140]
        img = np.dot(img[..., :3], b)
        return np.expand_dims(img, axis=-1)

    def t_grayscale_cv2(self, img):
        new_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
        for i, frame in enumerate(img):
            new_img[i] = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(new_img, axis=-1)

    def t_grayscale_mean(self, img):
        img = np.average(img, axis=-1)
        return img.astype(np.uint8)

    def t_normalize(self, img):
        return img.astype(np.float32) / 255

    def t_normalize_signed(self, img):
        return ((img / 255.) * 2 - 1).astype(np.float32)

    def t_gaussian_noise(self, img):
        return random_noise(img, mode='gaussian', var=self.augmentations.gn_var)

    def t_salt_and_pepper(self, img):
        return random_noise(img, mode='s&p', amount=self.augmentations.salt_and_pepper_amount)

    def t_speckle(self, img):
        return random_noise(img, mode='speckle', var=self.augmentations.speckle_var)

    def t_resize(self, img, target_length, target_height, target_width):
        return resize(img, (target_length, target_height, target_width), mode='constant', cval=0, preserve_range=True,
                      anti_aliasing=False).astype(np.uint8)

    def calc_fphb(self, hr, fps):
        hbs = hr/60
        fphb = fps / hbs
        return fphb

    def t_pad_size(self, img):
        # Pad edges of frames
        if self.transforms.target_height > img.shape[1] or self.transforms.target_width > img.shape[2]:
            pad_sequence = ((0, 0),
                            (int((self.transforms.target_height - img.shape[1])/2),
                            int((self.transforms.target_height - img.shape[1])/2)),
                            (int((self.transforms.target_width - img.shape[2])/2),
                             int((self.transforms.target_width - img.shape[2])/2)),
                            (0, 0))
            img = np.pad(img, pad_width=pad_sequence)
        return img

    def t_loop_length(self, img):
        org_img = img
        while len(img) < self.transforms.target_length:
            if len(org_img) <= self.transforms.target_length - len(img):
                img = np.append(img, org_img, axis=0)
            else:
                img = np.append(img, org_img[0:self.transforms.target_length - len(img)], axis=0)
        return img.astype(np.uint8)

    def t_crop(self, img):
        # Crop edges of frames
        crop_sequence = [(0, 0), (0, 0), (0, 0), (0, 0)]
        if self.transforms.crop_sides:
            crop_sequence[1] = (int(img.shape[1] / 10), int(img.shape[1] / 10))
            crop_sequence[2] = (int(img.shape[2] / 20), int(img.shape[2] / 20))
        if self.transforms.crop_length and img.shape[0] > self.transforms.target_length:
            diff = img.shape[0] - self.transforms.target_length
            rand = randint(0, diff)
            crop_sequence[0] = (rand, diff - rand)
        return crop(img, crop_width=tuple(crop_sequence)).astype(np.uint8)

    def t_translate_v(self, video):
        t_len = int(np.random.normal(0, self.augmentations.translate_v_std_dev_pxl))

        video = video.transpose(3, 0, 1, 2)

        final_video = np.zeros(video.shape)
        for i, channel in enumerate(video):
            new_img = np.zeros(channel.shape)
            for j, frame in enumerate(channel):
                translated_frame = np.zeros(frame.shape)
                if t_len < 0:
                    translated_frame[0:t_len, :] = frame[-t_len:, :]
                elif t_len > 0:
                    translated_frame[t_len:, :] = frame[0:-t_len, :]
                else:
                    translated_frame = frame
                new_img[j] = translated_frame
            final_video[i] = new_img

        return final_video.transpose(1, 2, 3, 0)

    def t_translate_h(self, video):
        t_len = int(np.random.normal(0, self.augmentations.translate_h_std_dev_pxl))

        video = video.transpose(3, 0, 1, 2)

        final_video = np.zeros(video.shape)
        for i, channel in enumerate(video):
            new_img = np.zeros(channel.shape)
            for j, frame in enumerate(channel):
                translated_frame = np.zeros(frame.shape)
                if t_len < 0:
                    translated_frame[:, 0:t_len] = frame[:, -t_len:]
                elif t_len > 0:
                    translated_frame[:, t_len:] = frame[:, 0:-t_len]
                else:
                    translated_frame = frame
                new_img[j] = translated_frame
            final_video[i] = new_img
        return final_video.transpose(1, 2, 3, 0)

    def t_rotate(self, video):
        t_rotation = np.random.normal(0, self.augmentations.rotate_std_dev_degrees)

        video = video.transpose(3, 0, 1, 2)

        final_video = np.zeros(video.shape)
        for i, channel in enumerate(video):
            new_img = np.zeros(channel.shape)
            for j, frame in enumerate(channel):
                rotated_frame = rotate(frame, t_rotation, resize=False, mode='constant', cval=0, preserve_range=True)
                new_img[j] = rotated_frame
            final_video[i] = new_img
        return final_video.transpose(1, 2, 3, 0)

