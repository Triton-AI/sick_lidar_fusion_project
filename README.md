# sick_lidar_fusion_project
This project uses Python ROS2 to fuse the SICK LiDAR with a camera. It runs object detection using the camera images and get the depth of detected objects using LiDAR pointcloud.

## Install
### Dependencies
- Ubuntu 20.04 or 22.04 (Earlier versions may also work but not tested)
- ROS2 Humble or Iron (Other ROS2 releases should also work but not tested)
- ultralytics (For running YOLOv8)
- If using Luxonis's OAK cameras
  - depthai
  - blobconverter (If wants to run YOLO model on OAK camera's computer)
  - NOTE: If not using OAK cameras please comment out the import statements for depthai and blobconverter

### Build
```
# clone the repo
cd sick_lidar_fusion_project
sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

### Run Fusion
Launch the SICK LiDAR and make sure it is publishing the ```/cloud_unstructured_fullframe``` topic.
```
# cd into sick_lidar_fusion_project
source install/setup.bash
ros2 run sick_lidar_fusion fusion_node
```

## Config
There are config variables as global variables in `src/cam_lidar_fusion/cam_lidar_fusion/fusion_node.py`.

Please pay attention to "LiDAR and Camera Coordinate Frames Config," "YOLO Model Config," and "Camera Config" as they likely need to be changed.

### LiDAR and Camera Coordinate Frames Config
`R`: `(3,3) np array` rotation matrix from lidar to camera

`T`: `(3,) np array` lidar frame's position seen from the camera's frame

### Features Config
`camera_type`: `str` choose between WEBCAM and OAK_LR. WEBCAM really means any camera that can be opened by OpenCV's `VideoCapture` function. OAK_LR is Luxonis's OAK-D Long Range camera, which is what's being used in our project for SICK's 10K Challenge 2024.

`show_lidar_projections`: `bool` whether or not to draw lidar points on the output image

`lidar_projections_size`: `int` radius of the lidar points on the image

`use_ROS_camera_topic`: `bool` use ROS subscriber to get camera images

`img_topic_name`: `str` the topic name of the image we want to subscribe to

`show_fusion_result_opencv`: `bool` use cv2.imshow to show the fusion result

`run_yolo_on_camera`: `bool` ONLY FOR OAK CAMERAS, run the YOLO detection model on camera or not

`record_video`: `bool` whether record a video or not

`video_file_name`: `str (.avi)` name of the video being recorded

### YOLO Model Config
`yolo_model_path`: `str` the path to the .pt YOLOv8 trained model file

`yolo_config`: `str` ONLY FOR RUNNING YOLO MODEL ON OAK CAMERAS, the path to the .json YOLOv8 trained model file

`yolo_model`: `str` ONLY FOR RUNNING YOLO MODEL ON OAK CAMERAS, the path to the .blob YOLOv8 trained model file

### Camera Config
`K`: `(3,3) np array` the camera's intrinsic matrix

`img_size`: `array-like` size (# rows, # columns) of the camera output image

## YOLO Model Training
See [here](https://cloud-swordfish-3c8.notion.site/Object-Detection-0d8e28b57b9e4c5b8a0de89ef90a1c05) for detailed instruction on how to train a YOLOv8 object detection model and how to convert the .pt model to blob model if you want to run the YOLO model on an OAK camera.
- Please put the trained .pt YOLOv8 model in the `src/cam_lidar_fusion/model` folder.
- If you want to run YOLOv8 models on OAK cameras, please follow the instructions provided in the link above to convert the .pt file. Extract the result and put everything into the `src/cam_lidar_fusion/blob_model` folder. NOTE: We switched to 320x320 size instead of the 640x640 as shown in the link to reduce computation time of the YOLO model.

## Other scripts in cam_lidar_fusion
### camera_calibration.py
- Computes the `K` matrix for WEBCAM type camera. Will print the `K` matrix and the `img_size` in the console.
- Before running, please print the camera calibration grid
- To run, cd into its directory and `python3 camera_calibration.py` or run it in an IDE.
- Capture the grid using the camera and move the grid or the camera slowly.
- Press 'q' after a couple of seconds to finish recording the pictures for calibration.
- It will print the `K` matrix and the `img_size`, and put these values in the if statement for `self.camera_type == "WEBCAM"`.

### oak_d_camera_calibration.py
- Works the same way as camera_calibration.py but for OAK cameras.
- Can choose to set img_size to get `K` for that size
  - useful when we want to run YOLO on camera, where the output image size is currently forced to equal to the YOLO model's input img size.
  - to set custom image size, set `use_custom_img_size` to `True` and set the `custom_img_size` to desired value