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

- [ ] Add Unit Test
    - [x] ArUco detector
    - [ ] Camera systems
        - [x] Single realsense system
        - [x] Dual realsense system
        - [ ] FLIR camera system
    - [ ] UI controller
        - [x] `_get_starting_index()` is possible to write unit test
        - [ ] not for others though

## File Structure

| File | Usage |
| --- | --- |
| `aruco_detector/aruco_detector.py` | class for detecting ArUco markers |
| `camera_objects/camera_abstract_class.py` | base class for all kinds of camera |
| `camera_objects/realsense_camera_system.py` | derived class for Realsense cameras |
| `camera_objects/flir_camera_system.py` | derived class for FLIR cameras |
| `camera_config/GH3_camera_config.yaml` | config file for FLIR grasshopper3 cameras |
| `camera_config/ORYX_camera_config.yaml` | config file for FLIR ORYX cameras |
| `ui_objects/opencv_ui_controller.py` | main controller for UI |
| `unit_tests/test_aruco_detector.py` | unit test for aruco detector |
| `unit_tests/test_dual_realsense_system.py` | unit test for dual realsense camera system |
| `unit_tests/test_realsense_camera_system.py` | unit test for realsense camera system |
| `main_dual_realsense.py` | main function for starting application with dual realsense camera system |
| `main_flir.py` | main function for starting application with FLIR cameras |
| `main_realsense.py` | main function for starting application with realsense camera |
| `requirements.txt` | note for environment requirements |

> [!NOTE]
> config for ORYX cameras are custom made, as the trigger lines can be connected differently.

> [!WARNING]
> config for ORYX cameras are still untested, require further testing to make sure it works.

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
├── aruco_detector/
│   ├── aruco_detector.py
├── camera_objects/
│   ├── camera_abstract_class.py
│   └── realsense_camera_system.py
│   └── flir_camera_system.py
├── camera_config/
│   ├── GH3_camera_config.yaml
│   ├── ORYX_camera_config.yaml
├── ui_objects/
│   ├── opencv_ui_controller.py
├── unit_tests/
│   ├── test_aruco_detector.py
│   ├── test_dual_realsense_system.py
│   ├── test_realsense_camera_system.py
├── main_dual_realsense.py
├── main_flir.py
├── main_realsense.py
├── requirements.txt
└── README.md
```