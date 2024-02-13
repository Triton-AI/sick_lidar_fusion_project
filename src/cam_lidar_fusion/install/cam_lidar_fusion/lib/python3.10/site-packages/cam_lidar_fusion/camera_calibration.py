# Computes the Intrinsic Parameters of the camera
# Maps Camera frame 3D points onto Image pixel frame 2D positions
# x_pixel = K @ X_camera

import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Chessboard configuration
rows = 9    # Number of corners (not cells) in row
cols = 6    # Number of corners (not cells) in column
size = 21  # Physical size of a cell (mm) (the distance between neighboring corners).

# Input images capturing the chessboard above
# input_files = '../data/chessboard/*.jpg'

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (size,0,0), (2*size,0,0) ....,((cols-1)*size,(rows-1)*size,0)
# cols and rows flipped because cols represent x and rows represent y
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = (np.mgrid[0:cols, 0:rows] * size).T.reshape(-1, 2)
# print(objp)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Capture Video for calibration
cap = cv2.VideoCapture(0)
print("Please move the camera slowly when you are collecting sample images")
print("Please press 'q' to quit once you are done collecting sample images for calibration")
while True:
    key = cv2.waitKey(300) & 0xFF
    # if 'q' is pressed, break from the loop
    if key == ord("q"):
        break
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    found, corners = cv2.findChessboardCorners(gray_frame, (cols, rows), None)

    # If found, add object points, image points (after refining them)
    if found:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray_frame, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(frame, (cols, rows), corners2, found)
        cv2.imshow('frame', frame)

cv2.destroyAllWindows()
cap.release()
print("# of Images taken (want at least 10): ", len(objpoints))

ret, K, d, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_frame.shape[::-1], None, None)

print('K: ', K)
print('d: ', d)
print('img size: ', gray_frame.shape)

# # To undistort an image: ##########################################################################################
# img = cv2.imread('your_img.jpg')
# h, w = img.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 1, (w, h))
#
# # undistort
# dst = cv2.undistort(img, K, d, None, newcameramtx)
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.png', dst)

# ####################################################################################################################

# Visualize undistort results ######################################################################################
# cap = cv2.VideoCapture(0)
# while True:
#     key = cv2.waitKey(300) & 0xFF
#     # if 'q' is pressed, break from the loop
#     if key == ord("q"):
#         break
#     ret, frame = cap.read()
#
#     h, w = frame.shape[:2]
#     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 1, (w, h))
#
#     # undistort
#     undistort_frame = cv2.undistort(frame, K, d, None, newcameramtx)
#
#     # crop the image
#     x, y, w, h = roi
#     undistort_frame = undistort_frame[y:y+h, x:x+w]
#
#     cv2.imshow('Original', frame)
#     cv2.imshow('Undistorted', undistort_frame)
#
# cv2.destroyAllWindows()
# cap.release()
