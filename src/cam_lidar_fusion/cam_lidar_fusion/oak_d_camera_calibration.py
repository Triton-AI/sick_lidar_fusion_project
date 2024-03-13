# See depthai hello world example: https://docs.luxonis.com/projects/api/en/latest/tutorials/hello_world/

import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets

# Oak-d Camera initialization #######################################################################3

# Start defining a pipeline
pipeline = depthai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.create(depthai.node.ColorCamera)
# For the demo, just set a larger RGB preview size for OAK-D
cam_rgb.setBoardSocket(depthai.CameraBoardSocket.RGB)
cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)

# Create output
xout_rgb = pipeline.create(depthai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
# cam_rgb.video.link(xout_rgb.input)
cam_rgb.setPreviewSize(320, 320)
cam_rgb.preview.link(xout_rgb.input)

device = depthai.Device(pipeline)
q_rgb = device.getOutputQueue("rgb")

frame = None
detections = []

# Chessboard configuration ########################################################################
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

print("Please move the camera slowly when you are collecting sample images")
print("Please press 'q' to quit once you are done collecting sample images for calibration")
while True:
    key = cv2.waitKey(300) & 0xFF
    # if 'q' is pressed, break from the loop
    if key == ord("q"):
        break
    
    in_rgb = q_rgb.tryGet()
    if in_rgb is not None:
        frame = in_rgb.getCvFrame()

    if frame is not None:
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
print("# of Images taken (want at least 10): ", len(objpoints))

ret, K, d, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_frame.shape[::-1], None, None)

print('K: ', K)
print('d: ', d)
print('img size: ', gray_frame.shape)