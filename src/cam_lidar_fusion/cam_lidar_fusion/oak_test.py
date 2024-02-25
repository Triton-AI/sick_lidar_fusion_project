from oak_camera import OakDCamera
import cv2
cam = OakDCamera(width=1920, height=1080)

while True:
    in_rgb = cam.queue_xout.tryGet()
    if in_rgb is not None:
        frame = in_rgb.getCvFrame()
        cv2.imshow('frame', frame)
