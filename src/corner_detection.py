"""
# Copyright 2021. Dongwon Kim All rights reserved.
#
# File name : corner_detection.py
#
# Written by Dongwon Kim
#
# CV assignment1
#   corner detection, threshold, NMS with window
#
# Modificatoin history
#   written by Dongwon Kim on Oct 27, 2021
"""
"""
# Corner Detection
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os

from gaussian_filter import *

LENNA_PATH = '../images/lenna.png'
SHAPES_PATH = '../images/shapes.png'
lenna = cv2.imread(LENNA_PATH, cv2.IMREAD_GRAYSCALE)
shapes = cv2.imread(SHAPES_PATH, cv2.IMREAD_GRAYSCALE)

"""
# Apply Gaussian Filter
"""
gausFilter = get_gaussian_filter_2d(7, 1.5)
filteredLenna = cross_correlation_2d(lenna, gausFilter)
filteredShapes = cross_correlation_2d(shapes, gausFilter)

"""
# corner respose
"""
def compute_corner_response(img):
    sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Sx = cross_correlation_2d(img, sobelX)
    Sy = cross_correlation_2d(img, sobelY)

    uniform = np.ones(shape=(5, 5))
    Sxx = Sx ** 2
    Syy = Sy ** 2
    Sxy = np.multiply(Sx, Sy)

    Mxx = cross_correlation_2d(Sxx, uniform)
    Myy = cross_correlation_2d(Syy, uniform)
    Mxy = cross_correlation_2d(Sxy, uniform)

    detM = Mxx * Myy - Mxy ** 2
    traceM = Mxx + Myy
    R = detM - 0.04 * (traceM ** 2)

    R = np.where(R < 0, 0, R)
    maxi = R.max()
    
    normalizedR = R / maxi
    return normalizedR
"""
## Result
"""
startTime = time.perf_counter()
cornerLenna = compute_corner_response(lenna)
endTime = time.perf_counter()
print("<Lenna> corner response computation time: {}sec".format(endTime - startTime))
plt.axis('off')
fig1 = plt.gcf()
plt.imshow(cornerLenna, 'gray')
fig1.savefig('../result/part_3_corner_raw_lenna.png')
plt.show()

startTime = time.perf_counter()
cornerShapes = compute_corner_response(shapes)
endTime = time.perf_counter()
print("<Shapes> corner response computation time: {}sec".format(endTime - startTime))
plt.axis('off')
fig2 = plt.gcf()
plt.imshow(cornerShapes, 'gray')
fig2.savefig('../result/part_3_corner_raw_shapes.png')
plt.show()

"""
# Thresholding
"""
def threshold(img, response, threshold):
    for i in range(response.shape[0]):
        for j in range(response.shape[1]):
            if(response[i, j] > threshold):
                cv2.circle(img, (j, i), 2, (0, 255, 0), 2, -1)
    return img

lennac = cv2.cvtColor(lenna, cv2.COLOR_GRAY2BGR)
markedLenna = threshold(lennac, cornerLenna, 0.1)
plt.axis('off')
fig3 = plt.gcf()
plt.imshow(markedLenna)
fig3.savefig('../result/part_3_corner_bin_lenna.png')
plt.show()

shapesc = cv2.cvtColor(shapes, cv2.COLOR_GRAY2BGR)
markedShapes = threshold(shapesc, cornerShapes, 0.1)
plt.axis('off')
fig4 = plt.gcf()
plt.imshow(markedShapes)
fig4.savefig('../result/part_3_corner_bin_shapes.png')
plt.show()

"""
# NMS
"""
def non_maximum_suppression_win(R, winSize):
    threshold = 0.1
    height = R.shape[0]
    width = R.shape[1]
    kernel = np.zeros(shape=(winSize, winSize))
    paddedR = padding_2d(R, kernel)
    temp = np.zeros(shape=(winSize, winSize))

    for i in range(height):
        for j in range(width + winSize //2):
            temp = paddedR[i:i + winSize, j:j + winSize]
            maxi = temp.max()
            if(j >= width):
                break
            else:
                if(R[i, j] != maxi):
                    R[i, j] = 0
                elif(R[i, j] < threshold):
                    R[i, j] = 0
            
    return R

"""
## Result
"""
startTime = time.perf_counter()
supLenna = non_maximum_suppression_win(cornerLenna, 11)
endTime = time.perf_counter()
print("<Lenna> NMS win computation time: {}sec".format(endTime - startTime))
supmarkedLenna = threshold(lennac, supLenna, 0.0)
plt.axis('off')
fig5 = plt.gcf()
plt.imshow(supmarkedLenna)
fig5.savefig('../result/part_3_corner_sup_lenna.png')
plt.show()

startTime = time.perf_counter()
supShapes = non_maximum_suppression_win(cornerShapes, 11)
endTime = time.perf_counter()
print("<Shapes> NMS win computation time: {}sec".format(endTime - startTime))
supmarkedShapes = threshold(shapesc, supShapes, 0.0)
plt.axis('off')
fig6 = plt.gcf()
plt.imshow(supmarkedShapes)
fig6.savefig('../result/part_3_corner_sup_shapes.png')
plt.show()
