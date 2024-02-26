from oak_camera import OakDCamera
import cv2
import depthai as dai
# cam = OakDCamera(width=1920, height=1080)

# while True:
#     in_rgb = cam.queue_xout.get()
#     if in_rgb is not None:
#         frame = in_rgb.getCvFrame()
#         cv2.imshow('frame', frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break


pipeline = dai.Pipeline()
cam_rgb = pipeline.create(dai.node.ColorCamera)
# cam_rgb.setPreviewSize(1448, 568)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)

# with dai.Device(pipeline) as device:
#     q_rgb = device.getOutputQueue(name='rgb', maxSize=1, blocking=False)
#     while True:
#         frame = q_rgb.tryGet()
#         if frame is not None:
#             cv_frame = frame.getCvFrame()
#             cv2.imshow('frame', cv_frame)
        
#         key = cv2.waitKey(30) & 0xFF
#         if key == ord('q'):
#             break
    
#     cv2.destroyAllWindows()

device = dai.Device(pipeline)
q_rgb = device.getOutputQueue(name='rgb', maxSize=1, blocking=False)
while True:
    frame = q_rgb.tryGet()
    if frame is not None:
        cv_frame = frame.getCvFrame()
        cv2.imshow('frame', cv_frame)
    
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()


def get_oak_queue():
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    # cam_rgb.setPreviewSize(1448, 568)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name='rgb', maxSize=1, blocking=False)
        return q_rgb