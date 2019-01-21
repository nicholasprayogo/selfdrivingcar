import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import glob
import os
showstep = False


def show_img(img, showstep=showstep, wait=False, name="img"):
    if showstep == True:
        cv2.imshow(name, img)
    if wait == True:
        cv2.waitKey(0)


def adjust_original_image(img, adjustment):
    height = img.shape[0]
    imgOriginalAdjusted = img[:(height-adjustment), :, :]
    return imgOriginalAdjusted


def draw_on_original(imgOriginalAdjusted, left_line_pts, right_line_pts, vertices):
    img_warped = transform(imgOriginalAdjusted, vertices)
    cv2.fillPoly(img_warped, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(img_warped, np.int_([right_line_pts]), (0, 255, 0))
    blended = transform(img_warped, vertices, mode="dewarp")
    # after dewarp the lane lines, add lanelines to original image
    blended = cv2.addWeighted(imgOriginalAdjusted, 1., blended, 0.3, 0)
    return blended


def mask(img, adjustment=0):
    height = img.shape[0]
    width = img.shape[1]

    botleft = (0, (height-adjustment))
    # have to make clean mask so that perspective transform is successful
    topleft = (width/2 - 70, (height+adjustment)/2 + 70)
    topright = (width/2 + 70, (height+adjustment)/2 + 70)
    # + adjustment to compensate?
    botright = (width, (height-adjustment))
    vertices = np.array([[botleft, topleft, topright, botright]], dtype=np.int32)
    # img.shape[0]=height-adjustment
    # defining a blank mask to start with
    # need to resize image after masking

    img2 = img[:(height-adjustment), :]
    mask = np.zeros_like(img2)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img2.shape) > 2:
        channel_count = img2.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
        print(ignore_mask_color)
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img2, mask)
    # show_img(mask,name="mask2",showstep=True, wait=True)
    return vertices, masked_image, img2


def transform(undist, vertices, mode="warp"):
    img_size = (undist.shape[1], undist.shape[0])
    print(img_size)
    print(vertices)
    # botleft = vertices[0]
    # topleft = vertices[1]
    # topright = vertices[2]
    # botright = vertices[3]
    [[botleft, topleft, topright, botright]] = vertices
    # undistcopy = np.copy(undist) # always draw on copy
    # imgwithcorners = cv2.drawChessboardCorners(undistcopy, (nx,ny), corners, ret)
    src = np.float32(vertices)
    width1 = np.sqrt((topright[0]-topleft[0])**2+(topright[1]-topleft[1])**2)
    width2 = np.sqrt((botright[0]-botleft[0])**2+(botright[1]-botleft[1])**2)
    max_width = max(int(width1), int(width2))

    height1 = np.sqrt((topright[0]-botright[0])**2+(topright[1]-botright[1])**2)
    height2 = np.sqrt((botright[0]-topright[0])**2+(botleft[1]-topleft[1])**2)
    max_height = max(int(height1), int(height2))

    # dst = np.float32([[offset,img_size[1]], [offset, 0],
    #                          [img_size[0]-offset, 0],
    #                          [img_size[0]-offset, img_size[1]]])

    dst = np.float32([[0, max_height], [0, 0], [max_width, 0], [max_width, max_height]])
    # have to make clean mask so that perspective transform is successful
    M = cv2.getPerspectiveTransform(src, dst)
    if mode == "warp":
        #warped = cv2.warpPerspective(undist, M, (max_width+100,max_height+100))
        warped = cv2.warpPerspective(undist, M, img_size)
        #show_img(warped, showstep = True, wait = False)
        return warped

    elif mode == "dewarp":
        dewarped = cv2.warpPerspective(undist, M, img_size, flags=cv2.WARP_INVERSE_MAP)
        return dewarped
