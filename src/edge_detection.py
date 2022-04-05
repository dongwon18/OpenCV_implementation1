"""
# Copyright 2021. Dongwon Kim All rights reserved.
#
# File name : edge_detection.py
#
# Written by Dongwon Kim
#
# Computer Vision
#   img gradient, NMS
#
# Modificatoin history
#   written by Dongwon Kim on Oct 27, 2021
"""

"""
# Edge detection
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
# Image Gradient
"""
def compute_image_gradient(img):
    sobelX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Sx = cross_correlation_2d(img, sobelX)
    Sy = cross_correlation_2d(img, sobelY)

    mag = np.zeros(img.shape, dtype=np.float64)
    direction = np.zeros(img.shape, dtype=np.float64)

    mag = np.sqrt(Sx ** 2 + Sy ** 2)
    direction = np.arctan2(Sy, Sx) + np.pi
    
    return mag, direction

startTime = time.perf_counter()
magLenna, dirLenna = compute_image_gradient(lenna)
endTime = time.perf_counter()
print("<Lenna> computation time: {}sec".format(endTime - startTime))
plt.axis('off')
fig1 = plt.gcf()
plt.imshow(magLenna, 'gray')
fig1.savefig('../result/part_2_edge_raw_lenna.png')
plt.show()

startTime = time.perf_counter()
magShapes, dirShapes = compute_image_gradient(shapes)
endTime = time.perf_counter()
print("<Shapes> computation time: {}sec".format(endTime - startTime))
plt.axis('off')
fig2 = plt.gcf()
plt.imshow(magShapes, 'gray')
fig2.savefig('../result/part_2_edge_raw_shapes.png')
plt.show()

"""
# NMS
"""
def non_maximum_suppression_dir(mag, dir):
    degreeMove = np.array([
        [0, -1, 0 ,1], 
        [-1, 1, 1, -1],
        [-1, 0, 1, 0],
        [-1, -1, 1, 1]
    ])
    height = dir.shape[0]
    width = dir.shape[1]
    quanDir, indices = quantize(dir)

    for i in range(height):
        for j in range(width):
            index = int(indices[i, j])
            if(i == 0 and j == 0):
                # [0,0] no candidate for 45 degre
                if(index == 0):
                    dy = 0
                    dx = 1                    
                elif(index == 2):
                    dy = 1
                    dx = 0
                elif(index == 3):
                    dy = 0
                    dx = 1
                candi1 = quanDir[i+dy, j + dx]
                maxi = max(candi1, quanDir[i, j])
                if(quanDir[i, j] != maxi):
                    mag[i, j] = 0 
            # [0, end] no candidate for 135 degree
            elif(i == 0 and j == width - 1):
                if(index == 0):
                    dy = 0
                    dx = -1                    
                elif(index == 1):
                    dy = 1
                    dx = -1
                elif(index == 2):
                    dy = 1
                    dx = 0
                candi1 = quanDir[i+dy, j + dx]
                maxi = max(candi1, quanDir[i, j])
                if(quanDir[i, j] != maxi):
                    mag[i, j] = 0
            
            # [end, 0] no cadidate for 135 degree
            elif(i == height -1 and j == 0):
                if(index == 0):
                    dy = 0
                    dx = 1                    
                elif(index == 1):
                    dy = -1
                    dx = 1
                elif(index == 2):
                    dy = -1
                    dx = 0
                candi1 = quanDir[i+dy, j + dx]
                maxi = max(candi1, quanDir[i, j])
                if(quanDir[i, j] != maxi):
                    mag[i, j] = 0
            
            # [end, end] no candidate for 45 degree
            elif(i == height -1 and j == width - 1):
                if(index == 0):
                    dy = 0
                    dx = -1                    
                elif(index == 2):
                    dy = -1
                    dx = 0
                elif(index == 3):
                    dy = -1
                    dx = -1
                candi1 = quanDir[i+dy, j + dx]
                maxi = max(candi1, quanDir[i, j])
                if(quanDir[i, j] != maxi):
                    mag[i, j] = 0

            # two candidate for 0 degree
            elif(i == 0):
                if(index == 0):
                    dy = degreeMove[index, 0]
                    dx = degreeMove[index, 1]
                    candi1 = quanDir[i + dy, j + dx]
                    dy = degreeMove[index, 2]
                    dx = degreeMove[index, 3]
                    candi2 = quanDir[i + dy, j + dx] 
                    maxi = max(candi1, quanDir[i, j], candi2)
                    if(quanDir[i, j] != maxi):
                        mag[i, j] = 0
                else:
                    if(index == 1):
                        dy = -1
                        dx = 1                    
                    elif(index == 2):
                        dy = 1
                        dx = 0
                    elif(index == 3):
                        dy = 1
                        dx = 1              
                
                    candi1 = quanDir[i+dy, j + dx]
                    maxi = max(candi1, quanDir[i, j])
                    if(quanDir[i, j] != maxi):
                        mag[i, j] = 0
            # two candidate for degree 0
            elif(i == height -1):
                if(index == 0):
                    dy = degreeMove[index, 0]
                    dx = degreeMove[index, 1]
                    candi1 = quanDir[i + dy, j + dx]
                    dy = degreeMove[index, 2]
                    dx = degreeMove[index, 3]
                    candi2 = quanDir[i + dy, j + dx] 
                    maxi = max(candi1, quanDir[i, j], candi2)
                    if(quanDir[i, j] != maxi):
                        mag[i, j] = 0
                else:
                    if(index == 1):
                        dy = -1
                        dx = 1                    
                    elif(index == 2):
                        dy = -1
                        dx = 0
                    elif(index == 3):
                        dy = -1
                        dx = -1              
                
                    candi1 = quanDir[i+dy, j + dx]
                    maxi = max(candi1, quanDir[i, j])
                    if(quanDir[i, j] != maxi):
                        mag[i, j] = 0
            # two candidate for degree 90
            elif(j == 0):
                if(index == 2):
                    dy = degreeMove[index, 0]
                    dx = degreeMove[index, 1]
                    candi1 = quanDir[i + dy, j + dx]
                    dy = degreeMove[index, 2]
                    dx = degreeMove[index, 3]
                    candi2 = quanDir[i + dy, j + dx] 
                    maxi = max(candi1, quanDir[i, j], candi2)
                    if(quanDir[i, j] != maxi):
                        mag[i, j] = 0
                else:
                    if(index == 0):
                        dy = 0
                        dx = 1                    
                    elif(index == 1):
                        dy = -1
                        dx = 1
                    elif(index == 3):
                        dy = 1
                        dx = 1              
                
                    candi1 = quanDir[i+dy, j + dx]
                    maxi = max(candi1, quanDir[i, j])
                    if(quanDir[i, j] != maxi):
                        mag[i, j] = 0
            
            # two candidate for degree 90
            elif(j == width - 1):
                if(index == 2):
                    dy = degreeMove[index, 0]
                    dx = degreeMove[index, 1]
                    candi1 = quanDir[i + dy, j + dx]
                    dy = degreeMove[index, 2]
                    dx = degreeMove[index, 3]
                    candi2 = quanDir[i + dy, j + dx] 
                    maxi = max(candi1, quanDir[i, j], candi2)
                    if(quanDir[i, j] != maxi):
                        mag[i, j] = 0
                else:
                    if(index == 0):
                        dy = 0
                        dx = -1                    
                    elif(index == 1):
                        dy = 1
                        dx = -1
                    elif(index == 3):
                        dy = -1
                        dx = -1              
                
                    candi1 = quanDir[i+dy, j + dx]
                    maxi = max(candi1, quanDir[i, j])
                    if(quanDir[i, j] != maxi):
                        mag[i, j] = 0
            else:
                dy = degreeMove[index, 0]
                dx = degreeMove[index, 1]
                candi1 = quanDir[i + dy, j + dx]
                dy = degreeMove[index, 2]
                dx = degreeMove[index, 3]
                candi2 = quanDir[i + dy, j + dx] 
                maxi = max(candi1, quanDir[i, j], candi2)
                if(quanDir[i, j] != maxi):
                    mag[i, j] = 0

    return mag

def quantize(dir):
    index = np.zeros(shape = dir.shape)
    for i in range(dir.shape[0]):
        for j in range(dir.shape[1]):
            dir[i, j] = dir[i, j] // 45
            index[i, j] = dir[i, j] % 4
            dir[i, j] = dir[i, j] * 45
    return dir, index

"""
# Result
"""
"""
## Lenna
"""
startTime = time.perf_counter()
supLenna = non_maximum_suppression_dir(magLenna, dirLenna)
endTime = time.perf_counter()
print("<Lenna> NMS compupation time: {}sec".format(endTime - startTime))
plt.axis('off')
fig3 = plt.gcf()
plt.imshow(supLenna, 'gray')
fig3.savefig('../result/part_2_edge_sup_lenna.png')
plt.show()

"""
## Shape
"""
startTime = time.perf_counter()
supShape = non_maximum_suppression_dir(magShapes, dirShapes)
endTime = time.perf_counter()
print("<Shapes> NMS compupation time: {}sec".format(endTime - startTime))
plt.axis('off')
fig4 = plt.gcf()
plt.imshow(supShape, 'gray')
fig4.savefig('../result/part_2_edge_sup_shapes.png')
plt.show()