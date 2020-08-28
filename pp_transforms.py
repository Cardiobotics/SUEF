import numpy as np
import cv2

def apply_transforms(data, flags):
    '''
    Called from preprocessing and calls appropriate transforms, regulated by flags.
    :param data: data from dicom file (Dict)
    :param flags: transform flags (Dict)
    :return: The transformed pixel_array from the dicom dict.
    '''
    img = data.pixel_array
    org_fps = data.RecommendedDisplayFrameRate

    if flags['gs']:
        img = rgb2gray(img)
    return img

def rgb2gray(img):
    '''
    Converts a 3-channel RGB image to grayscale.
    :param img: image data in array form (np.arr or python arr)
    :return: The greyscale img in numpy.array form.
    '''
    # Luminence numbers for converting RGB to grayscale
    b = [0.2989, 0.5870, 0.1140]
    gray_img = np.dot(img[...,:3], b)
    gray_img = np.expand_dims(gray_img, axis=0)
    return gray_img.astype(np.uint8)
