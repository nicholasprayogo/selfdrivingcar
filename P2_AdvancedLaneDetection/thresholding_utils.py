import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import glob
import os
from misc_utils import show_img, showstep

nx = 9
ny = 6
#dir = os.path.dirname(os.path.abspath(__file__))
# img = cv2.imread(dir+"/signs_vehicles_xygrad.png")


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    print(thresh_min, thresh_max)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    abs_sobel = np.absolute(sobel)
    # Take the absolute value of the output from cv2.Sobel()
    print("normalized:", 255*abs_sobel/np.max(sobel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Normalize by dividing each entry of abs_sobel with max of it
    # Multiply by 255 to have 8-bit integer (would result in scale 0-255)

    print("abs_sobel: ", abs_sobel, "scaled: ", scaled_sobel)
    gradbinary = np.zeros_like(scaled_sobel)
    gradbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255
    print("binary", gradbinary)
    show_img(gradbinary)
    return gradbinary


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    magbinary = np.zeros_like(gradmag)
    magbinary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255
    show_img(magbinary)
    # Return the binary image
    return magbinary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # range of arctan2 is pi/2
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dirbinary = np.zeros_like(absgraddir)
    dirbinary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255

    # Return the binary images
    show_img(dirbinary)
    return dirbinary


def hls_select(img, thresh=(0, 255), thresh2=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]
    hlsbinary = np.zeros_like(s_channel)
    hlsbinary[(s_channel > thresh[0]) & (s_channel <= thresh[1]) & (
        l_channel > thresh2[0]) & (l_channel <= thresh[1])] = 255
    show_img(hlsbinary)
    return hlsbinary


# use this to debug
# for i in range(20,150,5):
#     abs_sobel_thresh(img, thresh_min=i, thresh_max=255)

# for i in range(5,150,5):
#     mag_thresh(img, sobel_kernel=3, mag_thresh=(i, 255))
#     print(i)
# at 50 start to get less noisier lane lines

# for i in range(10,150,5):
#     hls_binary = hls_select(img, thresh=(i, 255))

# mag_thresh(img)
# dir_threshold(img)
