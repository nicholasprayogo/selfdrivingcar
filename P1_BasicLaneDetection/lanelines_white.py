import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import math
# import threading

showstep = False
# SET TO TRUE IF WANT TO SHOW STEPS

rho = 1
theta = np.pi/180
threshold = 50
min_line_len = 100
max_line_gap = 160

dir_path = os.path.dirname(os.path.realpath(__file__))
# dir = dir_path + "/test_images"
# img = dir + "/solidWhiteCurve.jpg"
# imgOriginal = cv2.imread(img)
# print(imgOriginal)
# print(imgOriginal.shape)
# UNCOMMENT IF WANT TO TEST ON IMAGES

# def main():
#     while True:
#         ret, imgOriginal = cap.read()
#         img = canny(gaussian_blur(grayscale(imgOriginal), 5), 50, 150)
#         masked = roi(img, vertices)
#         line_img = hough_lines(masked, rho, theta, threshold, min_line_len, max_line_gap)
#         weighted_img = weighted_img(line_img, imgOriginal)
#         cv2.imshow("Weighted Image", weighted_img)
#         if cv2.waitKey(1) == ord("q"):
#             break
#
#     cv2.destroyAllWindows()

def showfig(image, colormap):
    print("start")
    plt.imshow(image, cmap=colormap)
    plt.show()

def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if showstep:
        cv2.imshow("gray", gray)
        cv2.waitKey(0)
    return gray


def gaussian_blur(img, kernel_size):
    blur = cv2. GaussianBlur(img, (kernel_size, kernel_size), 0)
    if showstep:
        cv2.imshow("blur", blur)
        cv2.waitKey(0)
    return blur


def canny(img, low_threshold, high_threshold):
    can = cv2.Canny(img, low_threshold, high_threshold)
    if showstep:
        cv2.imshow("Can", can)
        cv2.waitKey(0)

    # t = threading.Thread(target=showfig, name="canny", args=(can, 'gray'))
    # t.start()
    # plt.show()

    return can


def roi(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
        print(ignore_mask_color)
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    if showstep:
        cv2.imshow("mask", mask)
        cv2.waitKey(0)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    if showstep:
        cv2.imshow("mask", masked_image)
        cv2.waitKey(0)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, imgOriginal, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    # # Testing for optimal threshold value
    # for i in range(50, threshold, 5):
    #     print(i)
    #     lines = cv2.HoughLinesP(img, rho, theta, i, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #     line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #     draw_lines(line_img, lines)
    #     plt.imshow(line_img, cmap='gray')
    #     plt.show()

    # testing for optimal max line gap value
#for i in range(50, max_line_gap, 5):
    #print(i)
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    line_img = np.copy(imgOriginal)
    draw_lines(line_img, lines)
    # matplotlib is not thread friendly
    # t = threading.Thread(target=showfig, name="hough {}".format(i), args=(img, None))
    # t.start()
    if showstep:
        cv2.imshow("lines", line_img)
        cv2.waitKey(0)

    return line_img

def weighted_img(img, initial_img, α=0.5, β=0.5, γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def main():
    dir_vid = dir_path + "/test_videos"
    vid = dir_vid + "/solidWhiteRight.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(vid)
    ret, imgOriginal = cap.read()
    height = imgOriginal.shape[0]
    width = imgOriginal.shape[1]

    c1 = (0, height)
    c2 = (width/2 - 100, height/2 + 50)
    c3 = (width/2 + 100, height/2 + 50)
    c4 = (width, height)
    vertices = np.array([[c1, c2, c3, c4]], dtype=np.int32)
    # important to make it array of array, and dtype 32 int, because will need for polyfit

    out = cv2.VideoWriter(dir_path + '/test_videos_output/solidWhiteOut.mp4',fourcc, 30.0, (width, height))

    while cap.isOpened():
        ret, imgOriginal = cap.read()
        img = canny(gaussian_blur(grayscale(imgOriginal), 5), 50, 150)
        masked = roi(img, vertices)
        line_img = hough_lines(masked, imgOriginal, rho, theta, threshold, min_line_len, max_line_gap)
        weighted = weighted_img(line_img, imgOriginal)
        cv2.imshow("Weighted Image", weighted)
        out.write(weighted)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()

main()
