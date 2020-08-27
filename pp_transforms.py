import numpy as np

def apply_transforms(data, flags):
    img = data.pixel_array
    org_fps = data.RecommendedDisplayFrameRate

    if flags['gs']:
        img = rgb2gray(img)
    return img

def rgb2gray(img):
    # Luminence numbers for converting RGB to grayscale
    b = [0.2989, 0.5870, 0.1140]
    gray_img = np.dot(img[...,:3], b)
    gray_img = np.expand_dims(gray_img, axis=0)
    return gray_img.astype(np.uint8)
