import numpy as np


def rerange(image):
    # map CT value to (-100, 700), equal to level 300, window 800
    new_min = -100
    new_max = 700
    new_image = np.clip(image, new_min, new_max)
    new_image = (new_image + 100)/800 
    return new_image


def roi_crop_pad(image, center, bbox_size):

    bbox_size = [bbox_size, bbox_size, bbox_size]

    bbminx = int(center[0] - bbox_size[0] // 2)
    bbminy = int(center[1] - bbox_size[1] // 2)
    bbminz = int(center[2] - bbox_size[2] // 2)

    bbmaxx = bbminx + bbox_size[0]
    bbmaxy = bbminy + bbox_size[1]
    bbmaxz = bbminz + bbox_size[2]

    pad_xl, pad_yl, pad_zl = 0, 0, 0
    pad_xr, pad_yr, pad_zr = 0, 0, 0

    if bbminx < 0:
        pad_xl = -bbminx
        bbminx = 0

    if bbminy < 0:
        pad_yl = -bbminy
        bbminy = 0

    if bbminz < 0:
        pad_zl = -bbminz
        bbminz = 0

    if bbmaxx > image.shape[0]:
        pad_xr = bbmaxx - image.shape[0]
        bbmaxx = image.shape[0]

    if bbmaxy > image.shape[1]:
        pad_yr = bbmaxy - image.shape[1]
        bbmaxy = image.shape[1]

    if bbmaxz > image.shape[2]:
        pad_zr = bbmaxz - image.shape[2]
        bbmaxz = image.shape[2]

    image = image[bbminx:bbmaxx, bbminy:bbmaxy, bbminz:bbmaxz]

    image = np.pad(image, ((pad_xl, pad_xr), (pad_yl, pad_yr), (pad_zl, pad_zr)), mode='constant', constant_values=np.min(image))

    return image