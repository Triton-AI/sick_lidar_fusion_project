import cv2
import numpy as np
import matplotlib.pyplot as plt
# import torch

# Generate point cloud for testing #########################################################################
deg2rad = np.pi/180

# point cloud resolution angle list in rad
phi = np.array([-10, 0, 10]) * deg2rad
theta = np.arange(-25, 26, 2) * deg2rad

THETA, PHI = np.meshgrid(theta, phi)

num_row = len(phi)
num_col = len(theta)
# Generate random distance points for testing
r = np.random.randn(num_row, num_col) + 3000  # distances in m


# Convert lidar point cloud from polar coordinate to cartesian coordinate ##########################################
def polar2cart(r, THETA, PHI):
    # r, THETA, PHI are all size of (len(phi), len(theta))
    # Assumes theta = 0 is x-axis direction, z-axis points upward
    if not (r.shape == THETA.shape and r.shape == PHI.shape):
        print('Error: Input array shape does not match')
        print('shape r: ', r.shape)
        print('shape THETA: ', THETA.shape)
        print('shape PHI: ', PHI.shape)
        return
    XY = r * np.exp(1j * THETA) * np.cos(PHI)  # real part is x and img part is y
    X = XY.real
    Y = XY.imag
    Z = r * np.sin(PHI)
    return X.flatten(), Y.flatten(), Z.flatten(), r.flatten()


x_lidar, y_lidar, z_lidar, depth = polar2cart(r, THETA, PHI)
# stack x y z into 4 x n homogeneous matrix
points_lidar = np.vstack((x_lidar, y_lidar))
points_lidar = np.vstack((points_lidar, z_lidar))
ones = np.ones((len(x_lidar), ))
points_lidar = np.vstack((points_lidar, ones))  # (x_lidar, y_lidar, z_lidar, 1)

# Transform point cloud expressed in LiDAR frame to Camera frame ##############################################
R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
T = np.array([0, 0, 0])

lidar2cam = np.hstack((R, T.reshape((-1, 1))))
lidar2cam = np.vstack((lidar2cam, np.array([0, 0, 0, 1])))

points_cam = lidar2cam @ points_lidar  # 4 x n matrix, (x_cam, y_cam, z_cam, 1)
z_cam = points_cam[2, :]

# Project point cloud onto Image frame ######################################################################
# [Guess a focal length. Need to change when find accurate focal length of the camera]
# f = 0.11  # focal length in m
#
# # Compute image frame projection of point cloud in the camera frame
# camera2img = np.array([[f, 0, 0, 0], [0, f, 0, 0], [0, 0, 1, 0]])
# points_img = camera2img @ (points_cam / z_cam)  # 3 x n matrix, (x_img, y_img, 1)

# Convert camera frame points to pixel coordinate #################################################################
camera2pixel = np.array([[543.892, 0, 308.268], [0, 537.865, 214.227], [0, 0, 1]])  # K matrix from camera_calibration
img_size = np.array([480, 640])

points_pixel = camera2pixel @ (points_cam[0:3, :] / z_cam)  # 3 x n matrix, (x_pixel, y_pixel, 1)
points_pixel.astype(int)

# Create a depth matrix ###########################################################################################


def points_to_img(xs, ys, ps, size):
    coords = np.stack((ys, xs))
    abs_coords = np.ravel_multi_index(coords, size)
    img = np.bincount(abs_coords, weights=ps, minlength=size[0]*size[1])
    img = img.reshape(size)
    return img


x_pixel = points_pixel[0, :]
y_pixel = points_pixel[1, :]

depth_matrix = points_to_img(x_pixel, y_pixel, r, img_size)

def lidar2pixel(lidar_points, R, T, K):
    # lidar_points assumed to be 4xn np array, each col should be (x, y, z, i)
    # R and T are (3, 3) and (3, )-like np arrays
    # K is (3, 3) camera calibration matrix
    xyz_lidar = lidar_points[0:3, :]
    i = lidar_points[3, :]
    xyz_lidar = np.vstack((xyz_lidar, np.ones((len(i), ))))

    lidar2cam = np.hstack((R, T.reshape((-1, 1))))
    lidar2cam = np.vstack((lidar2cam, np.array([0, 0, 0, 1])))

    xyz_cam = lidar2cam @ xyz_lidar
    z_cam = xyz_cam[2, :]

    xy_pixel = (K @ (xyz_cam[0:3, :] / z_cam)).astype(int)

    return xy_pixel[0, :], xy_pixel[1, :], i  # return x_array, y_array, intensity_array


# Plotting #####################################################################################################

fig1 = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x_lidar, y_lidar, z_lidar)

fig2 = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(points_cam[0, :], points_cam[1, :], points_cam[2, :])

fig3 = plt.figure()
plt.scatter(points_pixel[0, :], points_pixel[1, :])
plt.show()
