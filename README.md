# Depth Estimator with ArUco Corner Detection

## Functionality
- [x] Read two Realsense IR camera streams and Realsense depth stream
- [x] Detect ArUco markers for each stream
- [x] Compute depths with detected ArUco markers and get depth from Realsense depth stream

### buttons for depth_estimator_streaming.py

- `s` or `S` to save the images
- `esc` to close program

### buttons for depth_estimator_images.py

- `l` or `L` to load the images
- `n` or `N` to show next image set
- `p` or `P` to show previous image set
- `esc` to close program

## Goal
- [ ] Refactor the code to include interfaces for FLIR cameras
    - [x] Interface for intel realsense cameras
    - [ ] Interface for FLIR cameras
- [x] Clean up the codes

## File Structure

| File | Usage |
| --- | --- |
| `opencv_ui_controller.py` | main controller for UI |
| `camera_objects/camera_abstract_class.py` | base class for all kinds of camera |
| `camera_objects/realsense_camera_system.py` | derived class for Realsense cameras |
| `aruco_detector.py` | class for detecting ArUco markers |
| `depth_estimator_images.py` | Load the results saved before |

```
.
├── Db/
│   └── Realsense_{Date}/
│       ├── depth_images/
│       │   ├── depth_image1.npy
│       │   ├── depth_image1.png
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
├── aruco_detector.py
├── opencv_ui_controller.py
├── depth_estimator_images.py
└── README.md
```