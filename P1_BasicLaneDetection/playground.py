import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


dir_path = os.path.dirname(os.path.realpath(__file__))
dir = dir_path + "/test_images"
img = dir + "/solidWhiteCurve.jpg"
print(img)

left_bottom = [0, 539]
right_bottom = [850, 539]
apex = [420, 200]

# Perform a linear fit (y=Ax+B) to each of the three sides of the triangle
# np.polyfit returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

print(fit_left, fit_right, fit_bottom)

a = 1+3
