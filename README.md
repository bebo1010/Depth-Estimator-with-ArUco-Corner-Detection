# Depth Estimator with ArUco Corner Detection

## Demo Video
1. Demo Video with Single Realsense Camera

| Object Distance | Depth from Stereo Camera | Depth from Depth Module |
|-----------------|--------------------------|-------------------------|
| 3803mm          | 3843mm (+40mm)           | 3906mm (+103mm)         |

![Single Realsense Example](https://github.com/user-attachments/assets/f7cd9f4e-2c18-45bd-abe9-7f3f4f8b4dd4)


## Environment Setup

1. Install Python 3.8 or higher.
2. Install the required packages:
    ```bash
    pip install -e .
    ```
3. Install the PySpin library:
    - Download the appropriate PySpin wheel file from [here](https://www.flir.com/products/spinnaker-sdk/)
    - [PySpin 4.0.0.116, Python 3.8](https://drive.google.com/file/d/1G4BkDU8xr4Tgu4M9vk-Q2gX3HwvO-WSZ/view?usp=sharing)
    - Install the wheel file:
        ```bash
        pip install <path_to_wheel_file>
        ```
4. Run program
    - To run with **single realsense** camera:
    ```python
    python -m src.main_realsense
    ```
    - To run with **dual realsense** camera:
    ```python
    python -m src.main_dual_realsense
    ```
    - To run with **dual FLIR** camera:
    ```python
    python -m src.main_flir
    ```

## Functionality
- [x] Detect ArUco markers from left and right image
    - [x] Compute depths with detected ArUco markers
- [x] Include interfaces for other cameras
    - [x] Interface for intel RealSense cameras
        - [x] Single RealSense camera
        - [x] Dual RealSense camera
    - [x] Interface for FLIR cameras
    - [x] Add support for multiple RealSense cameras
    - [x] Include interfaces for grabbing grayscale and RGB images
- [x] Display Functionality
    - [x] Horizontal lines and Vertical lines
    - [x] Epipolar lines
        - If **ArUco** are detected
            - Display epipolar lines from corner points
        - If **ArUco** are not detected
            - Display epipolar lines from key points of scene
            - The method for detecting key points defaults to `ORB`
                - Can be swapped to `SIFT`
    - [x] Freeze frame
    - [x] Information Panel
        - [x] 3D position of 4 corner points of ArUco
            - [x] Estimated 3D position
            - [x] RealSense 3D position
        - [x] Mouse hover 3D position
- [ ] Chessboard calibration for stereo camera
    - [x] Calibration and save image
    - [x] Load back the parameters and rectify the images
    - [ ] (Optional) Show reprojection error per image
- [x] Auto rectification with fundamental matrix
- [x] Saving the images
    - [x] Also save the detected points in disk
- [x] Load back the saved images
    - [x] Include camera parameters like focal length, baseline, etc
- [ ] Unit Tests
    - [x] ArUco detector
    - [x] Camera systems (not possible for FLIR cameras)
    - [x] Utility functions
    - [x] Epipolar line detector
    - [x] Chessboard Calibration
    - [x] File utility functions
    - [x] Display utility functions
    - [ ] Need to review all the unit tests at some time

### buttons for opencv_ui_controller.py

- `h` or `H` to show horizontal lines
- `v` or `V` to show vertical lines
- `e` or `E` to show epipolar lines
    - `n`, `N`, `p`, or `P` to change algorithm
- `s` or `S` to save the images
- `f` or `F` to freeze frame
- `a` or `A` to display detected ArUco marker
- `c` or `C` to toggle on calibration mode
    - `s` or `S` to save chessboard image
    - `c` or `C` to toggle off calibration mode and start calibration
- `l` or `L` to load back previous saved images
    - `n`, `N`, `p`, or `P` to change image pairs
- `esc` to close program

## Goal
- Finish auto document

## Auto Document Generating

1. Environment Setup
    ```bash
    pip install -e .[doc]
    ```
    - One time only
        - Quick start the documentation structure
        ```bash
        sphinx-quickstart docs
        ```
        - Generate ReStructured text files
        ```bash\
        cd docs
        sphinx-apidoc -o . ../src
        cd ..
        ```

2. Build the document pages
    - HTML
    ```bash
    sphinx-build -b html docs docs/_build/html
    ```

    - PDF file
    1. Install [Strawberry Perl](https://strawberryperl.com/) manually
    2. Install [MikTex](https://miktex.org/download)
    3. Check updates and install `latexmk` in `Miktex Console`
    ```bash
    cd docs
    make.bat latexpdf
    ```

3. Run local documentation server
    ```bash
    sphinx-autobuild docs docs/_build/html
    ```

## Note
> [!NOTE]
> config for ORYX cameras are custom made, as the trigger lines can be connected differently.

> [!WARNING]
> config for ORYX cameras are still untested, require further testing to make sure it works.

- To run linting check:
```bash
python -m pylint ./src/**/*.py --max-line-length=120 --disable=E1101,E0611,E0401,E0633,R0801 --max-args=10 --max-locals=30 --max-attribute=15
```
> [!NOTE]
> `E1101`: No member error. Suppressing this for opencv-python and pyrealsense2 packages.
> `E0611`: No name in module error.  Suppressing this for opencv-python and pyrealsense2 packages.
> `E0401`: Unable to import error. Suppressing this for unable to install PySpin on workflow dispatch.
> `E0633`: Unpacking non sequence error. Suppressing this for ArUco detector.
> `R0801`: Duplicate code between files warning. Suppressing this for main functions.
> `max-line-length`: Limits max characters per line.
> `max-args`: Limits max arguments for a function.
> `max-locals`: Limits max local variables within a function.
> `max-attribute`: Limits max instance attribute for a class.

- To run unit tests:
```bash
python -m unittest discover -s ./tests -v
```
> [!NOTE]
> discover all the unit tests in `./test` and run all tests