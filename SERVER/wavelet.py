import pywt
import cv2
import numpy as np


def wavelet(image, level=1, mode='haar'):
    img_array = image
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = np.float32(img_array)
    img_array /= 255
    coeffs = pywt.wavedec2(img_array, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    img_array_H = pywt.waverec2(coeffs_H, mode)
    img_array_H *= 255
    img_array_H = np.uint8(img_array_H)
    return img_array_H
