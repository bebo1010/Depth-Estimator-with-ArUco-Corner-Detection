# Depth Estimator with ArUco Corner Detection

## Environment Setup

1. Install Python 3.8 or higher.
2. Install the required packages:
    ```bash
    pip install .
    ```
3. Install the PySpin library:
    - Download the appropriate PySpin wheel file from [here](https://www.flir.com/products/spinnaker-sdk/)
    - [PySpin 4.0.0.116, Python 3.8](https://drive.google.com/file/d/1G4BkDU8xr4Tgu4M9vk-Q2gX3HwvO-WSZ/view?usp=sharing)
    - Install the wheel file:
        ```bash
        pip install <path_to_wheel_file>
        ```

## Functionality
- [x] Read two Realsense IR camera streams and Realsense depth stream
- [x] Detect ArUco markers for each stream
- [x] Compute depths with detected ArUco markers and get depth from Realsense depth stream
- [x] Include interfaces for other cameras
    - [x] Interface for intel realsense cameras
    - [x] Interface for FLIR cameras
    - [x] Add support for multiple realsense cameras
- [x] Unit Tests
    - [x] ArUco detector
    - [x] Camera systems (not possible for FLIR cameras)
    - [x] Utility functions

### buttons for opencv_ui_controller.py

- `h` or `H` to show horizontal lines
- `v` or `V` to show vertical lines
- `s` or `S` to save the images
- `esc` to close program

## Goal

Write another functionality to display epipolar line
Maybe use `e` to display or not
Could use SIFT or SURF to detect correspnding points

## File Structure

| File | Description |
| --- | --- |
| `src/aruco_detector/aruco_detector.py` | class for detecting ArUco markers |
| `src/camera_objects/single_camera/single_camera_system.py` | derived class for single camera systems |
| `src/camera_objects/single_camera/flir_camera_system.py` | derived class for FLIR cameras |
| `src/camera_objects/two_cameras/two_cameras_system.py` | base class for two camera systems |
| `src/camera_objects/two_cameras/dual_flir_system.py` | derived class for dual FLIR camera systems |
| `src/camera_objects/two_cameras/realsense_camera_system.py` | derived class for Realsense cameras |
| `src/camera_objects/two_cameras/dual_realsense_system.py` | derived class for dual Realsense camera systems |
| `src/camera_config/GH3_camera_config.yaml` | config file for FLIR grasshopper3 cameras |
| `src/camera_config/ORYX_camera_config.yaml` | config file for FLIR ORYX cameras |
| `src/ui_objects/opencv_ui_controller.py` | main controller for UI |
| `src/utils/file_utils.py` | utility functions for file operations |
| `src/main_dual_realsense.py` | main function for starting application with dual realsense camera system |
| `src/main_flir.py` | main function for starting application with FLIR cameras |
| `src/main_realsense.py` | main function for starting application with realsense camera |
| `tests/test_aruco_detector.py` | unit test for aruco detector |
| `tests/test_dual_realsense_system.py` | unit test for dual realsense camera system |
| `tests/test_realsense_camera_system.py` | unit test for realsense camera system |
| `tests/test_file_utils.py` | unit test for file utilities |
| `pyproject.toml` |configuration file for the project|

> [!NOTE]
> config for ORYX cameras are custom made, as the trigger lines can be connected differently.

> [!WARNING]
> config for ORYX cameras are still untested, require further testing to make sure it works.

```
.
├── src/
│   ├── aruco_detector/
│   │   ├── __init__.py
│   │   └── aruco_detector.py
│   ├── camera_config/
│   │   ├── GH3_camera_config.yaml
│   │   └── ORYX_camera_config.yaml
│   ├── camera_objects/
│   │   ├── __init__.py
│   │   ├── single_camera/
│   │   │   ├── __init__.py
│   │   │   ├── single_camera_system.py
│   │   │   └── flir_camera_system.py
│   │   └── two_cameras/
│   │       ├── __init__.py
│   │       ├── two_cameras_system.py
│   │       ├── dual_flir_system.py
│   │       ├── realsense_camera_system.py
│   │       └── dual_realsense_system.py
│   ├── Db/
│   │   └── {CameraType}_{Date}/
│   │       ├── depth_images/
│   │       │   ├── depth_image1_1.npy
│   │       │   ├── depth_image1_1.png
│   │       │   ├── depth_image2_1.npy # only appear if dual realsense are used
│   │       │   ├── depth_image2_1.png # only appear if dual realsense are used
│   │       │   └── ...
│   │       ├── left_images/
│   │       │   ├── left_image1.png
│   │       │   └── ...
│   │       ├── right_images/
│   │       │   ├── right_image1.png
│   │       │   └── ...
│   │       └── aruco_depth_log.txt
│   ├── ui_objects/
│   │   ├── __init__.py
│   │   └── opencv_ui_controller.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── file_utils.py
│   ├── main_dual_realsense.py
│   ├── main_flir.py
│   └── main_realsense.py
├── tests/
│   ├── test_aruco_detector.py
│   ├── test_dual_realsense_system.py
│   ├── test_file_utils.py
│   └── test_realsense_camera_system.py
├── README.md
└── pyproject.toml
````
