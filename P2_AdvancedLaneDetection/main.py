import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import glob
import os

from thresholding_utils import abs_sobel_thresh, mag_thresh, dir_threshold, hls_select
from misc_utils import show_img, showstep, adjust_original_image, draw_on_original, mask, transform
from lanedetect import sliding_window, fit_polynomial, fit_poly, search_around_poly

dir_path = os.path.dirname(os.path.abspath(__file__))
nx = 9
ny = 6
SCALAR_EBLUE = (255, 255, 102)


def measure_curvature_pixels(ploty, left_fit, right_fit):
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    yval = np.max(ploty)

    A = left_fit[0]
    B = left_fit[1]
    C = left_fit[2]

    d1_left = 2*A*yval + B
    d2_left = 2*A
    left_curverad = (1+(d1_left)**2)**(3/2)/abs(d2_left)

    D = right_fit[0]
    E = right_fit[1]
    F = right_fit[2]

    d1_right = 2*D*yval + E
    d2_right = 2*D
    right_curverad = (1+(d1_right)**2)**(3/2)/abs(d2_right)

    return left_curverad, right_curverad


def main():
    dir_vid = dir_path + "/test_videos"
    print(dir_vid)
    vid = dir_vid + "/harder_challenge_video.mp4"

    cap = cv2.VideoCapture(vid)
    print(cap.isOpened())
    while cap.isOpened():
        ret, img = cap.read()
        # cv2.imshow("original",img)
        gradbinary = abs_sobel_thresh(img, thresh_min=10, thresh_max=255)
        magbinary = mag_thresh(img, mag_thresh=(10, 255))
        # experiment with the combination of threshold values
        dirbinary = dir_threshold(img, sobel_kernel=3, thresh=(0.5, 1.3))
        hlsbinary = hls_select(img, thresh=(70, 255), thresh2=(0, 255))
        show_img(hlsbinary, showstep=False, name="hls")

        combined = np.zeros_like(dirbinary)
        #combined[(gradbinary == 255)&((magbinary == 255) & (dirbinary == 255))] = 255
        combined[(hlsbinary == 255) & (magbinary == 255)] = 255
        height_adjustment = 25
        vertices, masked_image, img2 = mask(combined, adjustment=height_adjustment)

        imgOriginalAdjusted = adjust_original_image(img, adjustment=height_adjustment)

        cv2.imshow("adjusted", imgOriginalAdjusted)

        show_img(masked_image, showstep=True, name="masked")
        # cv2.imshow("masked",masked_image)
        warped = transform(masked_image, vertices)

        histogram, leftx, lefty, rightx, righty, out_img = sliding_window(warped)

        out_img, left_fit, right_fit, ploty = fit_polynomial(warped)
        result, left_line_pts, right_line_pts = search_around_poly(warped, left_fit, right_fit)
        # plt.plot(histogram)
        # plt.show()
        # plt.pause(0.0005)
        show_img(out_img, name="sliding window", showstep=False)
        show_img(result, name="polyfit", showstep=True)

        dewarped = transform(result, vertices, mode="dewarp")
        show_img(dewarped, name="dewarped", showstep=False)

        blended = draw_on_original(imgOriginalAdjusted, left_line_pts, right_line_pts, vertices)

        left_curverad, right_curverad = measure_curvature_pixels(ploty, left_fit, right_fit)
        print("radius: ", left_curverad, right_curverad)

        cv2.putText(blended, "Left curvature: {}, Right curvature: {}".format(
            left_curverad, right_curverad), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SCALAR_EBLUE, 2)
        show_img(blended, name="blended", showstep=True)

        if cv2.waitKey(1) == ord("q"):
            break
            cap.release()
            cv2.destroyAllWindows()


plt.ion()
main()

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
