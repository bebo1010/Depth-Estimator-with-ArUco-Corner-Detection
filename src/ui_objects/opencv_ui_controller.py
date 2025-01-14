"""
Module for main UI controller.
"""
import os
from datetime import datetime
import logging
from typing import Tuple, Optional

import cv2
import numpy as np

from src.opencv_objects import ArUcoDetector, EpipolarLineDetector, ChessboardCalibrator
from src.camera_objects import TwoCamerasSystem
from src.utils import get_starting_index, setup_directories, setup_logging, save_images

class OpencvUIController():
    """
    UI controller for ArUco detection application.

    Functions:
        __init__(str, float, float, Tuple[int, int]) -> None
        set_camera_system(TwoCamerasSystem) -> None
        start() -> None
        _calibrate_cameras() -> None
        _display_image(np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str) -> None
        _draw_on_depth_image(np.ndarray, np.ndarray, Tuple[int, int]) -> np.ndarray
        _draw_on_gray_image(np.ndarray, int, Tuple[int, int], float) -> np.ndarray
        _get_starting_index(str) -> int
        _handle_key_presses(int, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]) -> bool
        _process_and_draw_chessboard(np.ndarray, np.ndarray) -> None
        _process_and_draw_images(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                 Optional[np.ndarray] = None, Optional[np.ndarray] = None) -> None
        _process_disparity_and_depth(np.ndarray, np.ndarray, Optional[np.ndarray] = None,
                                     Optional[Tuple[int, int]] = None
                                     ) -> Tuple[np.ndarray, float, float, float, Optional[float]]
        _save_chessboard_images(np.ndarray, np.ndarray) -> None
        _save_images(np.ndarray, np.ndarray, Optional[np.ndarray] = None, Optional[np.ndarray] = None) -> None
        _setup_window() -> None
        _update_window_title(bool) -> None
    """
    def __init__(self,
                 system_prefix: str,
                 focal_length: float,
                 baseline: float,
                 principal_point: Tuple[int, int]) -> None:
        """
        Initialize UI controller.

        args:
        No arguments.

        returns:
        No return.
        """
        self.base_dir = os.path.join("Db", f"{system_prefix}_{datetime.now().strftime('%Y%m%d')}")
        left_ir_dir = os.path.join(self.base_dir, "left_ArUco_images")
        left_chessboard_dir = os.path.join(self.base_dir, "left_chessboard_images")

        setup_directories(self.base_dir)
        self.image_index = get_starting_index(left_ir_dir)
        self.chessboard_image_index = get_starting_index(left_chessboard_dir) - 1

        setup_logging(self.base_dir)

        self.mouse_coords = {'x': 0, 'y': 0}
        self.window_size = (2000, 960)
        self.matrix_view_size = (1280, 960)
        self._setup_window()

        self.camera_system = None
        self.aruco_detector = ArUcoDetector()

        self.camera_params = {'focal_length': focal_length, 'baseline': baseline, 'principal_point': principal_point}

        self.display_option = {
            'horizontal_lines': False,
            'vertical_lines': False,
            'epipolar_lines': False,
            'display_aruco': False,
            'calibration_mode': False,
            'freeze_mode': False
        }

        self.epipolar_detector = EpipolarLineDetector()

        self.calibration_mode = False
        self.chessboard_calibrator = ChessboardCalibrator()
        # small chessboard pattern size
        self.chessboard_calibrator.pattern_size = (10, 7)
        self.chessboard_calibrator.square_size_mm = 10
        self.image_points = {'left': [], 'right': []}

    def set_camera_system(self, camera_system: TwoCamerasSystem) -> None:
        """
        Set the camera system for the application.

        args:
        No arguments.

        returns:
        No return.
        """
        self.camera_system = camera_system

    def start(self) -> None:
        """
        This method initializes the OpenCV window and enters a loop to continuously
        capture and process images from the camera system. It handles various key
        presses to perform actions such as saving images, toggling display options,
        and terminating the application.
        Args:
        No arguments.
        Returns:
        No return.
        """
        cv2.namedWindow("Combined View (2x2)")

        left_gray_image, right_gray_image = None, None
        first_depth_image, second_depth_image = None, None

        while True:
            self._update_window_title()

            if not self.display_option['freeze_mode']:
                success, left_gray_image, right_gray_image = self.camera_system.get_grayscale_images()
                _, first_depth_image, second_depth_image = self.camera_system.get_depth_images()
                if not success:
                    continue

            if self.display_option['calibration_mode']:
                self._process_and_draw_chessboard(left_gray_image, right_gray_image)
            else:
                matching_ids_result, matching_corners_left, matching_corners_right = \
                    self.aruco_detector.detect_aruco_two_images(left_gray_image, right_gray_image)
                self._process_and_draw_images(left_gray_image, right_gray_image,
                                              matching_ids_result, matching_corners_left, matching_corners_right,
                                              first_depth_image, second_depth_image)

            # Check for key presses
            key = cv2.pollKey() & 0xFF
            if self._handle_key_presses(key, left_gray_image, right_gray_image, first_depth_image, second_depth_image):
                break

    def _draw_aruco_rectangle(self, image, corners, marker_id):
        """
        Draw a rectangle from the 4 corner points with red color and display the marker ID.

        Args:
            image (np.ndarray): Image on which to draw the rectangle.
            corners (np.ndarray): Corner points of the ArUco marker.
            marker_id (int): ID of the ArUco marker.

        Returns:
            None.
        """
        logging.info("Drawing ArUco rectangle.")
        corners = corners.reshape((4, 2)).astype(int)  # Ensure corners are integers
        for i in range(4):
            start_point = tuple(corners[i])
            end_point = tuple(corners[(i + 1) % 4])
            cv2.line(image, start_point, end_point, (0, 0, 255), 2)

        # Add the marker ID at the top-left corner of the rectangle
        top_left_corner = tuple((corners[0][0], corners[0][1] - 10))
        cv2.putText(image, f"ID: {marker_id}", top_left_corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def _calibrate_cameras(self) -> None:
        """
        Calibrate the cameras using the saved chessboard images.

        Returns:
            None.
        """
        if self.image_points['left'] and self.image_points['right']:
            image_size = (self.camera_system.get_width(), self.camera_system.get_height())
            success = self.chessboard_calibrator.calibrate_stereo_camera(self.image_points['left'],
                                                                         self.image_points['right'],
                                                                         image_size)
            if success:
                logging.info("Stereo camera calibration successful.")
                self.chessboard_calibrator.save_parameters(self.base_dir)

        else:
            logging.warning("No chessboard images saved for calibration.")

    def _display_image(self,
                       left_colored: np.ndarray,
                       right_colored: np.ndarray,
                       first_depth_colormap: np.ndarray,
                       second_depth_colormap: np.ndarray,
                       aruco_info: str,
                       mouse_info: str) -> None:
        """
        Display the processed images on the window.

        Args:
            left_colored (np.ndarray): Colored image of the left camera.
            right_colored (np.ndarray): Colored image of the right camera.
            first_depth_colormap (np.ndarray): Color-mapped first depth image.
            second_depth_colormap (np.ndarray): Color-mapped second depth image.
            aruco_info (str): Information about detected ArUco markers.
            mouse_info (str): Information about mouse hover.

        Returns:
            None.
        """
        image_width, image_height = self.matrix_view_size

        top_row = np.hstack((left_colored, right_colored))
        bottom_row = np.hstack((first_depth_colormap, second_depth_colormap))
        combined_view = np.vstack((cv2.resize(top_row, (image_width, image_height // 2)),
                                   cv2.resize(bottom_row, (image_width, image_height // 2))))

        # Create a blank image with the desired window size
        window_image = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
        window_image.fill(255)  # White background
        window_image[:image_height, :image_width] = combined_view

        # Add ArUco and mouse information to the right side of the window
        x0 = 20
        y0, dy = 30, 30
        cv2.putText(window_image, "Units: mm", (image_width + x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)  # Black text
        for i, line in enumerate(aruco_info.split('\n')):
            y = y0 + (i + 1) * dy
            cv2.putText(window_image, line, (image_width + x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)  # Black text
        for i, line in enumerate(mouse_info.split('\n')):
            y = y0 + (i + len(aruco_info.split('\n')) + 1) * dy
            cv2.putText(window_image, line, (image_width + x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)  # Black text

        cv2.imshow("Combined View (2x2)", window_image)

    def _get_starting_index(self, directory: str) -> int:
        """
        Get the starting index for image files in the given directory.

        args:
            directory (str): The directory to search for image files.

        return:
            int:
                - int: The starting index for image files in the given directory.
        """
        if not os.path.exists(directory):
            return 1
        files = [f for f in os.listdir(directory) if f.endswith(".png")]
        indices = [
            int(os.path.splitext(f)[0].split("image")[-1])
            for f in files
        ]
        return max(indices, default=0) + 1

    def _handle_key_presses(self, key: int, left_gray_image: np.ndarray, right_gray_image: np.ndarray,
                            first_depth_image: Optional[np.ndarray], second_depth_image: Optional[np.ndarray]) -> bool:
        """
        Handle key presses for various actions.

        Args:
            key (int): Key code of the pressed key.
            left_gray_image (np.ndarray): Grayscale image of the left camera.
            right_gray_image (np.ndarray): Grayscale image of the right camera.
            first_depth_image (Optional[np.ndarray]): First depth image.
            second_depth_image (Optional[np.ndarray]): Second depth image.

        Returns:
            bool: True if the application should exit, False otherwise.
        """
        # Define actions for each key
        actions = {
            27: self._exit_program,  # ESC key
            ord('s'): lambda: self._save_images(left_gray_image, right_gray_image,
                                                first_depth_image, second_depth_image),
            ord('S'): lambda: self._save_images(left_gray_image, right_gray_image,
                                                first_depth_image, second_depth_image),
            ord('h'): lambda: self._toggle_option('horizontal_lines'),
            ord('H'): lambda: self._toggle_option('horizontal_lines'),
            ord('v'): lambda: self._toggle_option('vertical_lines'),
            ord('V'): lambda: self._toggle_option('vertical_lines'),
            ord('e'): lambda: self._toggle_option('epipolar_lines'),
            ord('E'): lambda: self._toggle_option('epipolar_lines'),
            ord('n'): self._next_detector,
            ord('N'): self._next_detector,
            ord('p'): self._previous_detector,
            ord('P'): self._previous_detector,
            ord('c'): self._toggle_calibration_mode,
            ord('C'): self._toggle_calibration_mode,
            ord('f'): self._toggle_freeze_mode,
            ord('F'): self._toggle_freeze_mode,
            ord('a'): lambda: self._toggle_option('display_aruco'),
            ord('A'): lambda: self._toggle_option('display_aruco'),
        }

        # Execute the corresponding action if the key is in the dictionary
        if key in actions:
            actions[key]()
            return key == 27  # Return True if the ESC key was pressed

        return False

    def _exit_program(self):
        logging.info("Program terminated by user.")
        self.camera_system.release()
        cv2.destroyAllWindows()

    def _save_images(self, left_gray_image, right_gray_image, first_depth_image, second_depth_image):
        if self.display_option['calibration_mode']:
            self._save_chessboard_images(left_gray_image, right_gray_image)
        else:
            save_images(self.base_dir, left_gray_image, right_gray_image,
                        self.image_index, first_depth_image, second_depth_image,
                        prefix="ArUco")
            self.image_index += 1

    def _toggle_option(self, option):
        self.display_option[option] = not self.display_option[option]
        self._update_window_title()

    def _next_detector(self):
        if self.display_option['epipolar_lines']:
            self.epipolar_detector.switch_detector('n')
            self._update_window_title()

    def _previous_detector(self):
        if self.display_option['epipolar_lines']:
            self.epipolar_detector.switch_detector('p')
            self._update_window_title()

    def _toggle_calibration_mode(self):
        self.display_option['calibration_mode'] = not self.display_option['calibration_mode']
        if self.display_option['calibration_mode']:
            self.display_option['freeze_mode'] = False
        if not self.display_option['calibration_mode']:
            self._calibrate_cameras()

    def _toggle_freeze_mode(self):
        self.display_option['freeze_mode'] = not self.display_option['freeze_mode']
        if self.display_option['freeze_mode']:
            self.display_option['calibration_mode'] = False

    def _process_and_draw_chessboard(self, left_gray_image: np.ndarray, right_gray_image: np.ndarray) -> None:
        """
        Process and draw chessboard corners on the images in calibration mode.

        Args:
            left_gray_image (np.ndarray): Grayscale image of the left camera.
            right_gray_image (np.ndarray): Grayscale image of the right camera.

        Returns:
            None.
        """
        # Define the scale factor
        scale_factor = 2

        # Downsample the images by the scale factor
        left_small = cv2.resize(left_gray_image,
                                (left_gray_image.shape[1] // scale_factor, left_gray_image.shape[0] // scale_factor))
        right_small = cv2.resize(right_gray_image,
                                 (right_gray_image.shape[1] // scale_factor, right_gray_image.shape[0] // scale_factor))

        # Detect chessboard corners on the downsampled images
        ret_left, corners_left_small = self.chessboard_calibrator.detect_chessboard_corners(left_small)
        ret_right, corners_right_small = self.chessboard_calibrator.detect_chessboard_corners(right_small)

        left_colored = cv2.cvtColor(left_gray_image, cv2.COLOR_GRAY2BGR)
        right_colored = cv2.cvtColor(right_gray_image, cv2.COLOR_GRAY2BGR)

        if ret_left and ret_right:
            # Rescale the corners back to the original image size
            corners_left = corners_left_small * scale_factor
            corners_right = corners_right_small * scale_factor

            left_colored = self.chessboard_calibrator.display_chessboard_corners(left_colored, corners_left)
            right_colored = self.chessboard_calibrator.display_chessboard_corners(right_colored, corners_right)

        self._display_image(left_colored, right_colored,
                            np.zeros_like(left_colored), np.zeros_like(right_colored),
                            aruco_info="", mouse_info="")

    def _process_and_draw_images(self,
                                 left_gray_image: np.ndarray,
                                 right_gray_image: np.ndarray,
                                 matching_ids_result: np.ndarray,
                                 matching_corners_left: np.ndarray,
                                 matching_corners_right: np.ndarray,
                                 first_depth_image: Optional[np.ndarray] = None,
                                 second_depth_image: Optional[np.ndarray] = None) -> None:
        """
        Process and draw on the images.

        Args:
            left_gray_image (np.ndarray): Grayscale image of the left camera.
            right_gray_image (np.ndarray): Grayscale image of the right camera.
            matching_ids_result (np.ndarray): Detected marker IDs.
            matching_corners_left (np.ndarray): Detected corner points of the left image.
            matching_corners_right (np.ndarray): Detected corner points of the right image.
            first_depth_image (Optional[np.ndarray]): First depth image.
            second_depth_image (Optional[np.ndarray]): Second depth image.

        Returns:
            None.
        """
        left_colored = cv2.cvtColor(left_gray_image, cv2.COLOR_GRAY2BGR)
        right_colored = cv2.cvtColor(right_gray_image, cv2.COLOR_GRAY2BGR)

        def draw_lines(image, step, orientation):
            for i in range(0, image.shape[0 if orientation == 'horizontal' else 1], step):
                if orientation == 'horizontal':
                    cv2.line(image, (0, i), (image.shape[1], i), (0, 0, 255), 1)
                else:
                    cv2.line(image, (i, 0), (i, image.shape[0]), (0, 0, 255), 1)

        def apply_colormap(depth_image):
            return np.zeros_like(left_colored) if depth_image is None else \
                cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        def calculate_3d_coords(xs, ys, depths):
            return [((x - self.camera_params['principal_point'][0]) * depth / self.camera_params['focal_length'],
                     (y - self.camera_params['principal_point'][1]) * depth / self.camera_params['focal_length'], depth)
                    for x, y, depth in zip(xs, ys, depths)]

        def update_aruco_info(marker_id,
                              estimated_3d_coords, realsense_3d_coords,
                              mean_depth_estimated, mean_depth_realsense):
            info = f"ArUco ID {marker_id}:\n"
            info += f"Estimated: ({estimated_3d_coords[0][0]:7.1f}, {estimated_3d_coords[0][1]:7.1f}, " \
                    f"{estimated_3d_coords[0][2]:7.1f}), ({estimated_3d_coords[1][0]:7.1f}, " \
                    f"{estimated_3d_coords[1][1]:7.1f}, {estimated_3d_coords[1][2]:7.1f})\n"
            info += f"            ({estimated_3d_coords[2][0]:7.1f}, {estimated_3d_coords[2][1]:7.1f}, " \
                    f"{estimated_3d_coords[2][2]:7.1f}), ({estimated_3d_coords[3][0]:7.1f}, " \
                    f"{estimated_3d_coords[3][1]:7.1f}, {estimated_3d_coords[3][2]:7.1f})\n"
            info += f"Mean Depth (Estimated): {mean_depth_estimated:7.2f}\n"
            info += f"RealSense: ({realsense_3d_coords[0][0]:7.1f}, {realsense_3d_coords[0][1]:7.1f}, " \
                    f"{realsense_3d_coords[0][2]:7.1f}), ({realsense_3d_coords[1][0]:7.1f}, " \
                    f"{realsense_3d_coords[1][1]:7.1f}, {realsense_3d_coords[1][2]:7.1f})\n"
            info += f"            ({realsense_3d_coords[2][0]:7.1f}, {realsense_3d_coords[2][1]:7.1f}, " \
                    f"{realsense_3d_coords[2][2]:7.1f}), ({realsense_3d_coords[3][0]:7.1f}, " \
                    f"{realsense_3d_coords[3][1]:7.1f}, {realsense_3d_coords[3][2]:7.1f})\n"
            info += f"Mean Depth (RealSense): {mean_depth_realsense:7.2f}\n\n"
            return info

        if self.display_option['horizontal_lines']:
            draw_lines(left_colored, 20, 'horizontal')
            draw_lines(right_colored, 20, 'horizontal')

        if self.display_option['vertical_lines']:
            draw_lines(left_colored, 20, 'vertical')
            draw_lines(right_colored, 20, 'vertical')

        first_depth_colormap = apply_colormap(first_depth_image)
        second_depth_colormap = apply_colormap(second_depth_image)

        aruco_info = ""
        for i, marker_id in enumerate(matching_ids_result):
            disparities, mean_disparity, variance_disparity, estimated_depth_mm, realsense_depth_mm = \
                self._process_disparity_and_depth(matching_corners_left[i], matching_corners_right[i],
                                                  first_depth_image)

            logging.info("Marker ID: %d, Calculated Depth: %.2f mm, Depth Image Depth: %s mm, "
                         "Mean Disparity: %.2f, Variance: %.2f, Disparities: %s",
                         marker_id, np.mean(estimated_depth_mm), np.mean(realsense_depth_mm),
                         mean_disparity, variance_disparity, disparities.tolist())

            # Calculate 3D coordinates
            estimated_3d_coords = calculate_3d_coords(
                matching_corners_left[i][:, 0], matching_corners_left[i][:, 1], estimated_depth_mm
            )
            realsense_3d_coords = calculate_3d_coords(
                matching_corners_left[i][:, 0], matching_corners_left[i][:, 1], realsense_depth_mm
            )

            aruco_info += update_aruco_info(marker_id,
                                            estimated_3d_coords, realsense_3d_coords,
                                            np.mean(estimated_depth_mm), np.mean(realsense_depth_mm))

        if self.display_option['epipolar_lines']:
            if len(matching_ids_result) > 0 and self.epipolar_detector.fundamental_matrix is not None:
                left_colored, right_colored = self.epipolar_detector.compute_epilines_from_corners(
                    left_colored, right_colored, matching_corners_left, matching_corners_right)
            else:
                left_colored, right_colored = self.epipolar_detector.compute_epilines_from_scene(
                    left_colored, right_colored)

        if self.display_option['display_aruco']:
            logging.info("Display ArUco option is enabled. Drawing rectangles.")
            for i, marker_id in enumerate(matching_ids_result):
                self._draw_aruco_rectangle(left_colored, matching_corners_left[i], marker_id)
                self._draw_aruco_rectangle(right_colored, matching_corners_right[i], marker_id)

        # Calculate mouse hover info
        mouse_x, mouse_y = self.mouse_coords['x'], self.mouse_coords['y']
        if first_depth_image is not None:
            scaled_x = int(mouse_x * (self.camera_system.get_width() / (self.matrix_view_size[0] // 2)))
            scaled_y = int(mouse_y * (self.camera_system.get_height() / (self.matrix_view_size[1] // 2)))

            depth_value = first_depth_image[scaled_y, scaled_x]

            mouse_x_3d = (scaled_x - self.camera_params['principal_point'][0]) \
                            * depth_value / self.camera_params['focal_length']
            mouse_y_3d = (scaled_y - self.camera_params['principal_point'][1]) \
                            * depth_value / self.camera_params['focal_length']
            mouse_info = f"Mouse: ({mouse_x_3d:7.1f}, {mouse_y_3d:7.1f}, {depth_value:7.1f})"
        else:
            mouse_info = "Mouse: (N/A, N/A, N/A)"

        self._display_image(left_colored, right_colored,
                            first_depth_colormap, second_depth_colormap,
                            aruco_info, mouse_info)

    def _process_disparity_and_depth(self,
                                     matching_corners_left: np.ndarray,
                                     matching_corners_right: np.ndarray,
                                     depth_image: Optional[np.ndarray] = None
                                     ) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
        """
        Calculate disparities, mean, variance, and depth from matching corners.

        Optionally include depth from depth image.

        Args:
            matching_corners_left (np.ndarray): Corner points of the left image.
            matching_corners_right (np.ndarray): Corner points of the right image.
            depth_image (Optional[np.ndarray]): Depth image for calculating depth from image (optional).

        Returns:
            Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
                - Disparities between matching corners.
                - Mean of disparities.
                - Variance of disparities.
                - Calculated depth per corner in mm.
                - Depths at the 4 corner points from depth image (if provided).
        """
        disparities = np.abs(matching_corners_left[:, 0] - matching_corners_right[:, 0])
        mean_disparity = np.mean(disparities)
        variance_disparity = np.var(disparities)

        estimated_depth_mm = (self.camera_params['focal_length'] * self.camera_params['baseline']) / disparities

        realsense_depth_mm = np.zeros_like(estimated_depth_mm)
        if depth_image is not None:
            for j, (cx, cy) in enumerate(matching_corners_left):
                realsense_depth_mm[j] = depth_image[min(max(int(cy), 0), depth_image.shape[0] - 1),
                                               min(max(int(cx), 0), depth_image.shape[1] - 1)]

        return disparities, mean_disparity, variance_disparity, estimated_depth_mm, realsense_depth_mm

    def _save_chessboard_images(self, left_gray_image: np.ndarray, right_gray_image: np.ndarray) -> None:
        """
        Save chessboard images to disk and store image points for calibration.

        Args:
            left_gray_image (np.ndarray): Grayscale image of the left camera.
            right_gray_image (np.ndarray): Grayscale image of the right camera.

        Returns:
            None.
        """
        ret_left, corners_left = self.chessboard_calibrator.detect_chessboard_corners(left_gray_image)
        ret_right, corners_right = self.chessboard_calibrator.detect_chessboard_corners(right_gray_image)

        if ret_left and ret_right:
            self.image_points['left'].append(corners_left)
            self.image_points['right'].append(corners_right)

            self.chessboard_image_index += 1
            save_images(self.base_dir, left_gray_image, right_gray_image,
                        self.chessboard_image_index, prefix="chessboard")

    def _setup_window(self) -> None:
        """
        Setup OpenCV window and set the mouse callback.

        args:
        No arguments.

        returns:
        No return.
        """
        def _mouse_callback(event, x, y, _flags, _param):
            """Update the mouse position."""
            if event == cv2.EVENT_MOUSEMOVE:
                if 0 <= y < self.matrix_view_size[1] // 2:
                    if 0 <= x < self.matrix_view_size[0] // 2:
                        self.mouse_coords['x'], self.mouse_coords['y'] = x, y
                    elif self.matrix_view_size[0] // 2 <= x < self.matrix_view_size[0]:
                        self.mouse_coords['x'], self.mouse_coords['y'] = x - self.matrix_view_size[0] // 2, y
                else:
                    self.mouse_coords['x'], self.mouse_coords['y'] = 0, 0

        # Create a window and set the mouse callback
        cv2.namedWindow("Combined View (2x2)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Combined View (2x2)", *self.window_size)
        cv2.setMouseCallback("Combined View (2x2)", _mouse_callback)

    def _update_window_title(self) -> None:
        """
        Update the window title with the current detector name if epipolar lines are shown.
        """
        if self.display_option['calibration_mode']:
            cv2.setWindowTitle("Combined View (2x2)",
                               f"Combined View (2x2) - Calibration Mode - Images Saved: {self.chessboard_image_index}")
        elif self.display_option['epipolar_lines']:
            detector_name = self.epipolar_detector.detectors[self.epipolar_detector.detector_index][0]
            cv2.setWindowTitle("Combined View (2x2)", f"Combined View (2x2) - Detector: {detector_name}")
        else:
            cv2.setWindowTitle("Combined View (2x2)", "Combined View (2x2)")
        if self.display_option['freeze_mode']:
            cv2.setWindowTitle("Combined View (2x2)", "Combined View (2x2) - Freeze Mode")
