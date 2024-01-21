import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
# from sensor_msgs.msg import PointCloud
from rclpy.qos import qos_profile_sensor_data
import numpy as np
# import matplotlib.pyplot as plt


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')
        self.lidar_subs_ = self.create_subscription(
            PointCloud2,
            '/cloud_unstructured_fullframe',
            self.lidar_subs_callback,
            qos_profile_sensor_data
        )
        self.lidar_subs_  # prevent unused variable warning
        self.depth_matrix = np.array([])

    def lidar_subs_callback(self, msg):
        # print("Received: ", msg.fields)  # Print the received message
        lidar_points = np.array(list(read_points(msg, skip_nans=True))).T  # 4xn matrix, (x,y,z,i)
        R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])  # Rotation from lidar to camera
        T = np.array([0, 0, 0])  # Translation from lidar to camera
        K = np.array([[543.892, 0, 308.268], [0, 537.865, 214.227], [0, 0, 1]])  # K matrix from camera_calibration
        xs, ys, ps = lidar2pixel(lidar_points, R, T, K)

        img_size = (480, 640)
        filtered_x, filtered_y, filtered_p = filter_points(xs, ys, ps, img_size)

        self.depth_matrix = points_to_img(filtered_x, filtered_y, filtered_p, img_size)

        # print('xs size: ', xs.shape)
        # print('filtered_x size: ', filtered_x.shape)
        print('depth_matrix size: ', self.depth_matrix.shape)
        print('depth matrix: ', self.depth_matrix)
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # X, Y = np.meshgrid(np.arange(0, img_size[1], 1), np.arange(0, img_size[0], 1))
        # ax.plot3D(X, Y, self.depth_matrix)

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

    # Filter out the points on the back of the camera (points with negative z_cam values)
    index = z_cam >= 0
    xyz_cam = xyz_cam[:, index]
    z_cam = xyz_cam[2, :]
    i = i[index]

    xy_pixel = (K @ (xyz_cam[0:3, :] / z_cam)).astype(int)

    return xy_pixel[0, :], xy_pixel[1, :], i  # return x_array, y_array, intensity_array

def filter_points(xs, ys, ps, img_size):
    # Filter out the points that are not captured inside the camera
    lb = np.array([0, 0])
    ub = np.array(img_size).reshape((-1, ))
    points = np.stack((ys, xs), axis=1)

    inidx = np.all(np.logical_and(lb <= points, points < ub), axis=1)
    # print("xs shape: ", xs.shape)
    filtered_xy = points[inidx]
    filtered_p = ps[inidx]
    # print("filtered p shape: ", filtered_p.shape)
    # print("filtered x shape: ", filtered_xy[:, 1].shape)
    # print("filtered y shape: ", filtered_xy[:, 0].shape)

    return filtered_xy[:, 1], filtered_xy[:, 0], filtered_p  # return filtered_x, filtered_y, filtered_p

def points_to_img(xs, ys, ps, size):
    coords = np.stack((ys, xs))
    abs_coords = np.ravel_multi_index(coords, size)
    img = np.bincount(abs_coords, weights=ps, minlength=size[0]*size[1])
    img = img.reshape(size)
    return img

## The code below is "ported" from 
# https://github.com/ros/common_msgs/tree/noetic-devel/sensor_msgs/src/sensor_msgs
import sys
from collections import namedtuple
import ctypes
import math
import struct
# from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField

_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    """
    Read points from a L{sensor_msgs.PointCloud2} message.

    @param cloud: The point cloud to read from.
    @type  cloud: L{sensor_msgs.PointCloud2}
    @param field_names: The names of fields to read. If None, read all fields. [default: None]
    @type  field_names: iterable
    @param skip_nans: If True, then don't return any point with a NaN value.
    @type  skip_nans: bool [default: False]
    @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
    @type  uvs: iterable
    @return: Generator which yields a list of values for each point.
    @rtype:  generator
    """
    assert isinstance(cloud, PointCloud2), 'cloud is not a sensor_msgs.msg.PointCloud2'
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data, math.isnan
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step

def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt    += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()