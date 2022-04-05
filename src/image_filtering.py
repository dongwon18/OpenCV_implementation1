"""
# Copyright 2021. Dongwon Kim All rights reserved.
#
# File name : image_filtering.py
#
# Written by Dongwon Kim
#
# Computer Vision
#   cross correlation, gaussian filter
#
# Modificatoin history
#   written by Dongwon Kim on Oct 27, 2021
"""

"""
# Cross correlation
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

LENNA_PATH = '../images/lenna.png'
SHAPES_PATH = '../images/shapes.png'

def padding_2d(img, kernel):
    width = len(img)
    height = len(img[0])

    padX = kernel.shape[0] // 2
    padY = kernel.shape[1] // 2

    padded_img = np.zeros(shape=(width + 2 * padX, height + 2 * padY), dtype = np.float64)

    padded_img[padX:width+padX, padY:height+padY] = img.copy()

    return padded_img

def padding_1d(img, kernel):
    height = len(img)
    width = len(img[0])

    # horizontal
    if(kernel.shape[0] == 1):
        pad = kernel.shape[1] // 2
        padded_img = np.zeros(shape=(height, width + 2*pad), dtype=np.float64)
        padded_img[:, pad:width+pad]=img.copy()
        return padded_img

    # vertical
    elif(kernel.shape[1]==1):
        pad = kernel.shape[0] // 2
        padded_img = np.zeros(shape=(height + 2 * pad, width), dtype=np.float64)
        padded_img[pad:height + pad, :]=img.copy()
        return padded_img

def cross_correlation_2d(img, kernel):
    width = len(img)
    height = len(img[0])
    padded_img = padding_2d(img, kernel)

    correlation = np.zeros(shape=(width, height), dtype = np.float64)

    kernelX = kernel.shape[0]
    kernelY = kernel.shape[1]

    for i in range(width):
        for j in range(height):
            correlation[i, j] = np.sum(np.multiply(kernel, padded_img[i:i + kernelX, j:j + kernelY]))


    return correlation

def cross_correlation_1d(img, kernel):
    width = len(img)
    height = len(img[0])
    padded_img = padding_1d(img, kernel)

    correlation = np.zeros(shape=(width, height), dtype = np.float64)

    kernelX = kernel.shape[0]
    kernelY = kernel.shape[1]

    # horizontal
    if(kernelX == 1):
        for i in range(width):
            for j in range(height):
                correlation[i, j] = np.sum(np.multiply(kernel, padded_img[i, j:j + kernelY]))
    # vertical
    elif(kernelY == 1):
        for i in range(width):
            for j in range(height):
                correlation[i, j] = np.sum(np.multiply(kernel, padded_img[i:i + kernelX, j:j+1]))

    return correlation

"""
# Gaussian Filter
"""
def get_gaussian_filter_1d(size, sigma):
    a = size // 2
    x = np.ogrid[-a:a+1]

    filter = np.exp(-1*(x*x) / (2*sigma*sigma))

    total_sum = filter.sum()

    filter = filter / total_sum

    filter = filter.reshape(len(filter), -1)

    return filter

def get_gaussian_filter_2d(size, sigma):
    a = size // 2
    y, x = np.ogrid[-a:a+1, -a:a+1]

    filter = np.exp(-1*(x*x + y*y) / (2*sigma*sigma))

    total_sum = filter.sum()

    filter = filter / total_sum

    return filter

"""
# Result
"""
kernel1D = get_gaussian_filter_1d(5,1)
print('kernel1D')
print(kernel1D)

kernel2D = get_gaussian_filter_2d(5, 1)
print('kernel2D')
print(kernel2D)


lenna = cv2.imread(LENNA_PATH, cv2.IMREAD_GRAYSCALE)
shapes = cv2.imread(SHAPES_PATH, cv2.IMREAD_GRAYSCALE)

sizes = [5, 11, 17]
sigmas = [1, 6, 11]

"""
## lenna
"""
fig = plt.figure(figsize=(30, 30))
for i in range(3):
    for j in range(3):
        gausFilter = get_gaussian_filter_2d(sizes[i], sigmas[j])
        filtered_img = cross_correlation_2d(lenna, gausFilter)
        ax = fig.add_subplot(3, 3, i*3 + j + 1)
        ax.imshow(filtered_img, 'gray')
        ax.axis('off')
        ax.set_title("{}*{}, sigma={}".format(sizes[i], sizes[i], sigmas[j]))
fig.savefig('../result/part_1_gaussian_filtered_lenna.png')


"""
## shapes
"""
fig = plt.figure(figsize=(30, 30))
for i in range(3):
    for j in range(3):
        gausFilter = get_gaussian_filter_2d(sizes[i], sigmas[j])
        filtered_img = cross_correlation_2d(shapes, gausFilter)
        ax = fig.add_subplot(3, 3, i*3 + j + 1)
        ax.imshow(filtered_img, 'gray')
        ax.axis('off')
        ax.set_title("{}*{}, sigma={}".format(sizes[i], sizes[i], sigmas[j]))
fig.savefig('../result/part_1_gaussian_filtered_shapes.png')

"""
## Comparing 1D and 2D
"""
"""
### Lenna
"""
vertical1D = get_gaussian_filter_1d(17, 6)
horizontal1D = vertical1D.T

startTime = time.perf_counter()
filteredImg1D = cross_correlation_1d(lenna, vertical1D)
filteredImg1D = cross_correlation_1d(filteredImg1D, horizontal1D)
endTime = time.perf_counter()
print("<Lenna> sequential 1D computational time: {}sec".format(endTime - startTime))

filter2D = get_gaussian_filter_2d(17, 6)
startTime = time.perf_counter()
filteredImg2D = cross_correlation_2d(lenna, filter2D)
endTime = time.perf_counter()
print("<Lenna> 2D computational time: {}sec".format(endTime - startTime))
plt.show()

def get_pixel_diff(img1, img2):
    diff = np.subtract(img1, img2)
    absSum = np.sum(np.abs(diff))

    return diff, absSum

diff, absSum = get_pixel_diff(filteredImg1D, filteredImg2D)
plt.imshow(diff, 'gray')
print('<Lenna> abs Sum of difference: {}'.format(absSum))
plt.show()

"""
### shapes
"""
vertical1D = get_gaussian_filter_1d(17, 6)
horizontal1D = vertical1D.T

startTime = time.perf_counter()
filteredImg1D = cross_correlation_1d(shapes, vertical1D)
filteredImg1D = cross_correlation_1d(filteredImg1D, horizontal1D)
endTime = time.perf_counter()
print("<Shapes> sequential 1D computational time: {}sec".format(endTime - startTime))

filter2D = get_gaussian_filter_2d(17, 6)
startTime = time.perf_counter()
filteredImg2D = cross_correlation_2d(shapes, filter2D)
endTime = time.perf_counter()
print("<Shapes> 2D computational time: {}sec".format(endTime - startTime))
plt.show()

diff, absSum = get_pixel_diff(filteredImg1D, filteredImg2D)
plt.imshow(diff, 'gray')
print('<Shapes> abs Sum of difference: {}'.format(absSum))
plt.show()