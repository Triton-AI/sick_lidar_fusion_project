import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2
# from sensor_msgs.msg import PointCloud
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import matplotlib.pyplot as plt
import cv2
# import depthai as dai
from ultralytics import YOLO

from ament_index_python import get_package_share_directory
import os

# from cam_lidar_fusion.cone_detection import detect_cones


class FusionNode(Node):
    def __init__(self):
        self.R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])  # Rotation from lidar to camera
        # self.T = np.array([0, 0.2, 0])  # lidar frame position seen from the camera's frame
        self.T = np.array([-0.02, 0.13, -0.015])  # oak lr test

        self.camera_type = "OAK_LR"  # WEBCAM, OAK_WIDE, OAK_LR
        self.use_ROS_camera_topic = False  # use ROS subscriber to get camera images
        self.show_fusion_result_opencv = False  # use cv2.imshow to show the fusion result

        # yolo_model_path = os.path.join(get_package_share_directory("cam_lidar_fusion"), "model/obstacle_v2_320.pt")
        # yolo_model_path = os.path.join(get_package_share_directory("cam_lidar_fusion"), "model/shelf_picker_v2_320.pt")
        yolo_model_path = os.path.join(get_package_share_directory("cam_lidar_fusion"), "model/cones_best.pt")

        self.yolo_model = YOLO(yolo_model_path)
        try:
            self.yolo_model.to(device='cuda')
        except:
            print('cuda not avaliable, use cpu')
            self.yolo_model.to(device='cpu')

        try:
            import depthai as dai
        except ImportError:
            print('Cannot import depthai, set use_ROS_camera_topic to True')
            self.use_ROS_camera_topic = True

        if self.camera_type == "WEBCAM":
            # webcam #########################################################################################################
            self.K = np.array([[543.892, 0, 308.268], [0, 537.865, 214.227], [0, 0, 1]])  # K matrix from camera_calibration
            self.img_size = (480, 640)  # Size of the img captured by the camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Cannot open webcam")
                exit()
        elif self.camera_type == "OAK_WIDE":
            # oak-d pro wide #################################################################################################
            self.K = np.array([[1007.03765, 0, 693.05655], [0, 1007.59267, 356.9163], [0, 0, 1]])  # K matrix from camera_calibration
            self.img_size = [720, 1280]  # Size of the img captured by the camera
            if not self.use_ROS_camera_topic:
                pipeline = self.get_oak_pipeline()
                self.device = dai.Device(pipeline)
        elif self.camera_type == "OAK_LR":
            # oak-d LR ########################################################################################################
            self.K = np.array([[1147.15312, 0., 936.61046], [0., 1133.707, 601.71022], [0, 0, 1]])
            self.img_size = (1200, 1920)
            if not self.use_ROS_camera_topic:
                pipeline = self.get_oak_pipeline()
                self.device = dai.Device(pipeline)
        else:
            print("camera type not supported")
            exit()

        super().__init__('fusion_node')
        self.lidar_subs_ = self.create_subscription(
            PointCloud2,
            '/cloud_unstructured_fullframe',
            self.lidar_subs_callback,
            qos_profile_sensor_data
        )
        self.lidar_subs_  # prevent unused variable warning
        self.depth_matrix = np.array([])

        self.oak_d_cam_subs = self.create_subscription(
            Image,
            '/oak/rgb/image_rect',
            self.oak_d_cam_subs_callback,
            qos_profile_sensor_data
        )
        self.frame = None
        
        self.cone_hsv_lb = np.array([109, 83, 131])  # hsv threshold lower bound for detecting cones
        self.cone_hsv_ub = np.array([180, 255, 255])  # hsv threshold upper bound for detecting cones

        self.fusion_img_pubs_ = self.create_publisher(Image, 'camera/fused_img', 10)
        self.bridge = CvBridge()
        
    def oak_d_cam_subs_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # self.img_size = self.frame.shape
            print("setted img_size: ", self.img_size)
        except CvBridgeError as e:
            print(e)

    def lidar_subs_callback(self, msg):
        frame = None
        max_dist_thresh = 10  # the max distance used for color coding in visualization window.
        # print("Received: ", msg.fields)  # Print the received message
        lidar_points = np.array(list(read_points(msg, skip_nans=True))).T  # 4xn matrix, (x,y,z,i)
        xs, ys, ps = lidar2pixel(lidar_points, self.R, self.T, self.K)

        filtered_x, filtered_y, filtered_p = filter_points(xs, ys, ps, self.img_size)

        self.depth_matrix = points_to_img(filtered_x, filtered_y, filtered_p, self.img_size)

        # Visualization ####################################################################################

        if self.use_ROS_camera_topic:  # use ROS subscriber for oak cameras
            if self.frame is not None:
                frame = self.frame.copy()
        else:
            if self.camera_type == "WEBCAM":
                ret, frame = self.cap.read()
            else:  # using oak cameras
                oak_q = self.device.getOutputQueue(name='rgb', maxSize=1, blocking=False)
                in_q = oak_q.tryGet()
                if in_q is not None:
                    frame = in_q.getCvFrame()
                    # print("got frame")
        
        if frame is not None:
            # Draw circles for the lidar points
            for i in range(len(filtered_p)):
                color_intensity = int((filtered_p[i] / max_dist_thresh * 255).clip(0, 255))
                cv2.circle(frame, (filtered_x[i], filtered_y[i]), 4, (0,color_intensity, 255 - color_intensity), -1)
            
            detections_xyxyn = self.yolo_predict(frame)
            for detection in detections_xyxyn:
                x1 = int(detection[0] * self.img_size[1])
                y1 = int(detection[1] * self.img_size[0])
                x2 = int(detection[2] * self.img_size[1])
                y2 = int(detection[3] * self.img_size[0])
                object_depth = np.min(self.depth_matrix[y1:y2, x1:x2])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
                cv2.putText(frame, str(round(object_depth, 2)) + "m", (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 1, 255)
            
            # OpenCV detection ###############################################################################################
            # cone_detection_boxes = detect_cones(frame, self.cone_hsv_lb, self.cone_hsv_ub)
            # frame_cv_copy = frame.copy()
            # # Draw box for detected cones
            # # Print the lidar distance of the cone above the box
            # for box in cone_detection_boxes:
            #     x = box[0]
            #     y = box[1]
            #     w = box[2]
            #     h = box[3]
            #     cone_dist = np.min(self.depth_matrix[y:y+h, x:x+w])
            #     # print('cone dist: ', cone_dist)
            #     cv2.putText(frame_cv_copy, str(round(cone_dist, 2)) + "m", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 1, 255)
            #     cv2.rectangle(frame_cv_copy, (x, y), (x+w, y+h), (1, 255, 1), 3)
            # # cv2.imshow('OpenCV threshold detection: ', frame_cv_copy)
            # ################################################################################################################

            if self.show_fusion_result_opencv:
                cv2.imshow('YOLO detection: ', frame)
                cv2.waitKey(1)

            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.fusion_img_pubs_.publish(img_msg)

        # Debugging #############################################################################################3
        # cv2.imshow('depth img', ((self.depth_matrix*1000).clip(0, 255)).astype(np.int8))
        # print('xs size: ', xs.shape)
        # print('filtered points size: ', filtered_x.shape)
        # print('filtered p: ', filtered_p)
        # print('depth_matrix size: ', self.depth_matrix.shape)
        # print('max in depth_matrix: ', np.max(self.depth_matrix))
        # np.set_printoptions(threshold=np.inf)
        # print('depth matrix: ', self.depth_matrix)
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # X, Y = np.meshgrid(np.arange(0, img_size[1], 1), np.arange(0, img_size[0], 1))
        # ax.plot3D(X, Y, self.depth_matrix)
    
    def get_oak_pipeline(self):
        pipeline = dai.Pipeline()
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        # cam_rgb.setPreviewSize(1448, 568)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        if self.camera_type == "OAK_LR":
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
        elif self.camera_type == "OAK_WIDE":
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
        cam_rgb.setInterleaved(False)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.video.link(xout_rgb.input)

        return pipeline
    
    def yolo_predict(self, img):
        results = self.yolo_model.predict(source=img, save=False, save_txt=False)

        # bounding box params https://docs.ultralytics.com/modes/predict/#boxes
        box = results[0].boxes.cpu()
        xyxyn = box.xyxyn.numpy().reshape((-1,))
        detections_xyxyn = list(zip(*[iter(xyxyn)] * 4))
        # confidence = box.conf.numpy()

        return detections_xyxyn


# Helper functions #########################################################################################
def lidar2pixel(lidar_points, R, T, K):
    # lidar_points assumed to be 4xn np array, each col should be (x, y, z, i)
    # R and T are (3, 3) and (3, )-like np arrays
    # K is (3, 3) camera calibration matrix
    xyz_lidar = lidar_points[0:3, :]
    # i = lidar_points[3, :]
    i = np.linalg.norm(xyz_lidar, axis=0)
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
    """
    Save the points (ps) into a matrix at their given locations,
        locations where there's no given points will be saved as np.inf
    :param xs: list of x (col) locations for the points
    :param ys: list of y (row) locations for the points
    :param ps: values for the points
    :param size: (row, col) size of the matrix
    :return : result np matrix
    """
    coords = np.stack((ys, xs))
    abs_coords = np.ravel_multi_index(coords, size)
    img = np.bincount(abs_coords, weights=ps, minlength=size[0]*size[1])
    img = img.reshape(size)
    img[img == 0.0] = np.inf
    return img

def convex_hull_pointing_up(ch):

    points_above_center, points_below_center = [], []
    
    x, y, w, h = cv2.boundingRect(ch)
    aspect_ratio = w / h

    if aspect_ratio < 0.8:
        vertical_center = y + h / 2

        for point in ch:
            if point[0][1] < vertical_center:
                points_above_center.append(point)
            elif point[0][1] >= vertical_center:
                points_below_center.append(point)

        left_x = points_below_center[0][0][0]
        right_x = points_below_center[0][0][0]
        for point in points_below_center:
            if point[0][0] < left_x:
                left_x = point[0][0]
            if point[0][0] > right_x:
                right_x = point[0][0]

        for point in points_above_center:
            if (point[0][0] < left_x) or (point[0][0] > right_x):
                return False
    else:
        return False
        
    return True

def detect_cones(frame, hsv_lb, hsv_ub):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    img_thresh = cv2.inRange(frame_hsv, hsv_lb, hsv_ub)
    # cv2.imshow('threshold', img_thresh)

    kernel = np.ones((5, 5))
    img_thresh_opened = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('thresh opened', img_thresh_opened)

    img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 5)

    img_edges = cv2.Canny(img_thresh_blurred, 80, 160)

    contours, _ = cv2.findContours(np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # img_contours = np.zeros_like(img_edges)
    # cv2.drawContours(img_contours, contours, -1, (255,255,255), 2)
    # cv2.imshow('contours', img_contours)

    approx_contours = []
    for c in contours:
        approx = cv2.approxPolyDP(c, 10, closed = True)
        approx_contours.append(approx)
    # img_approx_contours = np.zeros_like(img_edges)
    # cv2.drawContours(img_approx_contours, approx_contours, -1, (255,255,255), 1)

    all_convex_hulls = []
    for ac in approx_contours:
        all_convex_hulls.append(cv2.convexHull(ac))
    # img_all_convex_hulls = np.zeros_like(img_edges)
    # cv2.drawContours(img_all_convex_hulls, all_convex_hulls, -1, (255,255,255), 2)

    convex_hulls_3to10 = []
    for ch in all_convex_hulls:
        if 3 <= len(ch) <= 10:
            convex_hulls_3to10.append(cv2.convexHull(ch))
    # img_convex_hulls_3to10 = np.zeros_like(img_edges)
    # cv2.drawContours(img_convex_hulls_3to10, convex_hulls_3to10, -1, (255,255,255), 2)

    cones = []
    bounding_rects = []
    for ch in convex_hulls_3to10:
        if convex_hull_pointing_up(ch):
            cones.append(ch)
            rect = cv2.boundingRect(ch)
            bounding_rects.append(rect)
    # img_cones = np.zeros_like(img_edges)
    # cv2.drawContours(img_cones, cones, -1, (255,255,255), 2)
    # cv2.drawContours(img_cones, bounding_rects, -1, (1,255,1), 2)
    # cv2.imshow('find cones', img_cones)

    # img_res = frame
    # cv2.drawContours(img_res, cones, -1, (255,255,255), 2)

    # for rect in bounding_rects:
    #     cv2.rectangle(img_res, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (1, 255, 1), 3)
    # cv2.imshow('cone detection result', img_res)

    return bounding_rects


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

# ###############################################################################################################

def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()