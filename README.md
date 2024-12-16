# Depth Estimator with ArUco Corner Detection

## Functionality
- [x] Read two Realsense IR camera streams and Realsense depth stream
- [x] Detect ArUco markers for each stream
- [x] Compute depths with detected ArUco markers and get depth from Realsense depth stream
- [x] Include interfaces for other cameras
    - [x] Interface for intel realsense cameras
    - [x] Interface for FLIR cameras
    - [x] Add support for multiple realsense cameras

### buttons for opencv_ui_controller.py

- `h` or `H` to show horizontal lines
- `v` or `V` to show vertical lines
- `s` or `S` to save the images
- `esc` to close program

## Goal

To be added.....

## File Structure

| File | Usage |
| --- | --- |
| `ui_objects/opencv_ui_controller.py` | main controller for UI |
| `camera_objects/camera_abstract_class.py` | base class for all kinds of camera |
| `camera_objects/realsense_camera_system.py` | derived class for Realsense cameras |
| `camera_objects/flir_camera_system.py` | derived class for FLIR cameras |
| `camera_config/GH3_camera_config.yaml` | config file for FLIR grasshopper3 cameras |
| `aruco_detector.py` | class for detecting ArUco markers |
| `main_dual_realsense.py` | main function for starting application with two realsense cameras |
| `main_flir.py` | main function for starting application with FLIR cameras |
| `main_realsense.py` | main function for starting application with realsense camera |
| `requirements.txt` | note for environment requirements |

```
.
├── Db/
│   └── {Camera Type}_{Date}/
│       ├── depth_images/
│       │   ├── depth_image1_1.npy
│       │   ├── depth_image1_1.png
│       │   ├── depth_image2_1.npy # only appear if dual realsense are used
│       │   ├── depth_image2_1.png # only appear if dual realsense are used
│       │   └── ...
│       ├── left_images/
│       │   ├── left_image1.png
│       │   └── ...
│       ├── right_images/
│       │   ├── right_image1.png
│       │   └── ...
│       └── aruco_depth_log.txt
├── camera_objects/
│   ├── camera_abstract_class.py
│   └── realsense_camera_system.py
│   └── flir_camera_system.py
├── camera_config/
│   ├── GH3_camera_config.yaml
├── ui_objects/
│   ├── opencv_ui_controller.py
├── aruco_detector.py
├── main_dual_realsense.py
├── main_flir.py
├── main_realsense.py
├── requirements.txt
└── README.md
```