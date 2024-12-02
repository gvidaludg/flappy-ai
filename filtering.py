import numpy as np
import cv2
from scipy.signal import convolve2d

blur_kernel = (1 / 256.0) * np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
])


edge_kernel = np.array([
    [-1, -1, -1],
    [-1, +8, -1],
    [-1, -1, -1]
])


def bgra_convolve2d(img, kernel):
    blue = convolve2d(img[:, :, 0], kernel, 'valid')
    green = convolve2d(img[:, :, 1], kernel, 'valid')
    red = convolve2d(img[:, :, 2], kernel, 'valid')
    return np.stack([blue, green, red], axis=2).astype(np.uint8)


def from_buffer(buf, width, height):
    return np.asarray(bytearray(buf), dtype=np.uint8).reshape(height, width, 4)


def edge_detection(img):
    return bgra_convolve2d(img, edge_kernel)
