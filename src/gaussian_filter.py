"""
# Copyright 2021. Dongwon Kim All rights reserved.
#
# File name : gaussian_filter.py
#
# Written by Dongwon Kim
#
# Computer Vision
#   padding, gaussian filter, cross correlation
#
# Modificatoin history
#   written by Dongwon Kim on Oct 27, 2021
"""
import numpy as np

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