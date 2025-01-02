"""
Module for main UI controller.
"""
import os
from datetime import datetime
import logging
from typing import Tuple, Optional

import cv2
import numpy as np

from src.opencv_objects import ArUcoDetector, EpipolarLineDetector
from src.camera_objects import TwoCamerasSystem
from src.utils.file_utils import get_starting_index

class OpencvUIController():
    """
    UI controller for ArUco detection application.

    Functions:
        __init__(str, float, float) -> None
        set_camera_system(TwoCamerasSystem) -> None
        start() -> None
        _setup_directories() -> None
        _setup_logging() -> None
        _setup_window() -> None
        _process_disparity_and_depth(np.ndarray, np.ndarray,
                                     Optional[np.ndarray] = None, Optional[Tuple[int, int]] = None
                                     ) -> Tuple[np.ndarray, float, float, float, Optional[float]]
        _draw_on_gray_image(np.ndarray, int,
                            Tuple[int, int], float) -> np.ndarray
        _draw_on_depth_image(np.ndarray, np.ndarray, Tuple[int, int]) -> np.ndarray
        _display_image(np.ndarray, np.ndarray,
                       np.ndarray, np.ndarray, np.ndarray,
                       Optional[np.ndarray] = None) -> None
        _save_images(np.ndarray, np.ndarray, np.ndarray) -> None
    """
    def __init__(self, system_prefix: str, focal_length: float, baseline: float) -> None:
        """
        Initialize UI controller.

        args:
        No arguments.

        returns:
        No return.
        """
        self.base_dir = os.path.join("Db", f"{system_prefix}_{datetime.now().strftime('%Y%m%d')}")
        left_ir_dir = os.path.join(self.base_dir, "left_images")

        self._setup_directories()
        self.image_index = get_starting_index(left_ir_dir)

        self._setup_logging()

        self.mouse_x = 0
        self.mouse_y = 0
        self._setup_window()

        self.camera_system = None
        self.aruco_detector = ArUcoDetector()

        self.focal_length = focal_length
        self.baseline = baseline

        self.draw_horizontal_lines = False
        self.draw_vertical_lines = False

        self.epipolar_detector = EpipolarLineDetector()
        self.draw_epipolar_lines = False

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
        Start the application.
        - Press `s` or `S` to save images.
        - Press `esc` to exit.

        args:
        No arguments.

        returns:
        No return.
        """
        cv2.namedWindow("Combined View (2x2)")
        self._update_window_title()

        while True:
            success, left_gray_image, right_gray_image = self.camera_system.get_grayscale_images()
            _, first_depth_image, second_depth_image = self.camera_system.get_depth_images()
            if not success:
                continue
            matching_ids_result, matching_corners_left, matching_corners_right = \
                self.aruco_detector.detect_aruco_two_images(left_gray_image, right_gray_image)
            self._display_image(left_gray_image, right_gray_image,
                                matching_ids_result, matching_corners_left, matching_corners_right,
                                first_depth_image, second_depth_image)

            # Check for key presses
            key = cv2.pollKey() & 0xFF
            if key == 27:  # ESC key
                logging.info("Program terminated by user.")
                self.camera_system.release()
                cv2.destroyAllWindows()
                break
            if key == ord('s') or key == ord('S'):  # Save images
                self._save_images(left_gray_image, right_gray_image, first_depth_image, second_depth_image)
            if key == ord('h') or key == ord('H'):  # Toggle horizontal lines
                self.draw_horizontal_lines = not self.draw_horizontal_lines
            if key == ord('v') or key == ord('V'):  # Toggle vertical lines
                self.draw_vertical_lines = not self.draw_vertical_lines
            if key == ord('e') or key == ord('E'):  # Toggle epipolar lines
                self.draw_epipolar_lines = not self.draw_epipolar_lines
                self._update_window_title()
            if self.draw_epipolar_lines:
                if key == ord('n'):  # Switch to next detector
                    self.epipolar_detector.switch_detector('n')
                    self._update_window_title()
                if key == ord('p'):  # Switch to previous detector
                    self.epipolar_detector.switch_detector('p')
                    self._update_window_title()

    def _update_window_title(self) -> None:
        """
        Update the window title with the current detector name if epipolar lines are shown.
        """
        if self.draw_epipolar_lines:
            detector_name = self.epipolar_detector.detectors[self.epipolar_detector.detector_index][0]
            cv2.setWindowTitle("Combined View (2x2)", f"Combined View (2x2) - Detector: {detector_name}")
        else:
            cv2.setWindowTitle("Combined View (2x2)", "Combined View (2x2)")

    def _setup_directories(self) -> None:
        """
        Make directories for storing images and logs.

        args:
        No arguments.

        returns:
        No return.
        """
        os.makedirs(self.base_dir, exist_ok=True)

        left_ir_dir = os.path.join(self.base_dir, "left_images")
        right_ir_dir = os.path.join(self.base_dir, "right_images")
        depth_dir = os.path.join(self.base_dir, "depth_images")

        os.makedirs(left_ir_dir, exist_ok=True)
        os.makedirs(right_ir_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

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

    def _setup_logging(self) -> None:
        """
        Setup logging for the application.

        args:
        No arguments.

        returns:
        No return.
        """
        log_path = os.path.join(self.base_dir, "aruco_depth_log.txt")
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

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
                self.mouse_x, self.mouse_y = x, y

        # Create a window and set the mouse callback
        cv2.namedWindow("Combined View (2x2)")
        cv2.setMouseCallback("Combined View (2x2)", _mouse_callback)

    def _process_disparity_and_depth(self,
                                     matching_corners_left: np.ndarray,
                                     matching_corners_right: np.ndarray,
                                     depth_image: Optional[np.ndarray] = None,
                                     center_coords: Optional[Tuple[int, int]] = None
                                     ) -> Tuple[np.ndarray, float, float, float, Optional[float]]:
        """
        Calculate disparities, mean, variance, and depth from matching corners.

        Optionally include depth from depth image.

        Args:
            matching_corners_left (np.ndarray): Corner points of the left image.
            matching_corners_right (np.ndarray): Corner points of the right image.
            depth_image (Optional[np.ndarray]): Depth image for calculating depth from image (optional).
            center_coords (Optional[Tuple[int, int]]): Center coordinates for depth image lookup (optional).

        Returns:
            Tuple[np.ndarray, float, float, float, Optional[float]]:
                - Disparities between matching corners.
                - Mean of disparities.
                - Variance of disparities.
                - Calculated depth in mm.
                - Depth in mm from depth image (if provided).
        """
        disparities = np.abs(matching_corners_left[:, 0] - matching_corners_right[:, 0])
        mean_disparity = np.mean(disparities)
        variance_disparity = np.var(disparities)

        depth_mm_calc = (self.focal_length * self.baseline) / mean_disparity if mean_disparity > 0 else 0

        depth_mm_from_image = None
        if depth_image is not None and center_coords is not None:
            x, y = center_coords
            depth_mm_from_image = depth_image[min(max(int(y), 0), depth_image.shape[0] - 1),
                                              min(max(int(x), 0), depth_image.shape[1] - 1)]

        return disparities, mean_disparity, variance_disparity, depth_mm_calc, depth_mm_from_image

    def _draw_on_gray_image(self,
                            image: np.ndarray,
                            marker_id: int,
                            center_coords: Tuple[int, int],
                            depth_mm_calc: float) -> np.ndarray:
        """
        Draw marker ID and calculated depth on the grayscale image.

        Args:
            image (np.ndarray): Grayscale image to draw on.
            marker_id (int): Detected marker ID.
            center_coords (Tuple[int, int]): Center coordinates of the marker.
            depth_mm_calc (float): Calculated depth in mm.

        Returns:
            np.ndarray: Grayscale image with drawn markers and depth.
        """
        cv2.putText(image, f"ID:{marker_id} Depth:{int(depth_mm_calc)}mm",
                    center_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return image

    def _draw_on_depth_image(self,
                              depth_image: np.ndarray,
                              depth_colormap: np.ndarray,
                              mouse_coords: Tuple[int, int]) -> np.ndarray:
        """
        Draw depth data on the depth image.

        Args:
            depth_image (np.ndarray): Depth image.
            depth_colormap (np.ndarray): Color-mapped depth image.
            mouse_coords (Tuple[int, int]): Mouse coordinates.

        Returns:
            np.ndarray: Depth image with drawn depth data.
        """
        mouse_x, mouse_y = mouse_coords
        scaled_x = int(mouse_x * (self.camera_system.get_width() / 640))
        scaled_y = int(mouse_y * (self.camera_system.get_height() / 480))

        depth_value = depth_image[scaled_y, scaled_x]

        cv2.circle(depth_colormap, (scaled_x, scaled_y), 5, (0, 255, 255), -1)
        text = f"Depth: {depth_value} mm"
        cv2.putText(depth_colormap, text, (scaled_x + 10, scaled_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return depth_colormap

    def _display_image(self,
                       left_gray_image: np.ndarray,
                       right_gray_image: np.ndarray,
                       matching_ids_result: np.ndarray,
                       matching_corners_left: np.ndarray,
                       matching_corners_right: np.ndarray,
                       first_depth_image: Optional[np.ndarray] = None,
                       second_depth_image: Optional[np.ndarray] = None) -> None:
        """
        Display the processed images on the window.

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

        if self.draw_horizontal_lines:
            draw_lines(left_colored, 20, 'horizontal')
            draw_lines(right_colored, 20, 'horizontal')

        if self.draw_vertical_lines:
            draw_lines(left_colored, 20, 'vertical')
            draw_lines(right_colored, 20, 'vertical')

        first_depth_colormap = np.zeros_like(left_colored) if first_depth_image is None else \
            cv2.applyColorMap(cv2.convertScaleAbs(first_depth_image, alpha=0.03), cv2.COLORMAP_JET)

        second_depth_colormap = np.zeros_like(left_colored) if second_depth_image is None else \
            cv2.applyColorMap(cv2.convertScaleAbs(second_depth_image, alpha=0.03), cv2.COLORMAP_JET)

        for i, marker_id in enumerate(matching_ids_result):
            center_coords = tuple(np.mean(matching_corners_left[i], axis=0).astype(int))
            disparities, mean_disparity, variance_disparity, depth_mm_calc, depth_mm_from_image = \
                self._process_disparity_and_depth(matching_corners_left[i], matching_corners_right[i],
                                                  first_depth_image, center_coords)

            left_colored = self._draw_on_gray_image(left_colored, marker_id, center_coords, depth_mm_calc)

            depth_mm_from_image = str(depth_mm_from_image) if depth_mm_from_image is not None else "N/A"
            logging.info("Marker ID: %d, Calculated Depth: %.2f mm, Depth Image Depth: %s mm, "
                         "Mean Disparity: %.2f, Variance: %.2f, Disparities: %s",
                         marker_id, depth_mm_calc, depth_mm_from_image,
                         mean_disparity, variance_disparity, disparities.tolist())

        if self.draw_epipolar_lines:
            if len(matching_ids_result) > 0 and self.epipolar_detector.fundamental_matrix is not None:
                left_colored, right_colored = self.epipolar_detector.compute_epilines_from_corners(
                    left_colored, right_colored, matching_corners_left, matching_corners_right)
            else:
                left_colored, right_colored = self.epipolar_detector.compute_epilines_from_scene(
                    left_colored, right_colored)

        if 0 <= self.mouse_y < 480:
            if 0 <= self.mouse_x < 640:
                depth_coord = (self.mouse_x, self.mouse_y)
            elif 640 <= self.mouse_x < 1280:
                depth_coord = (self.mouse_x - 640, self.mouse_y)
            else:
                depth_coord = (0, 0)

            if first_depth_image is not None:
                first_depth_colormap = self._draw_on_depth_image(first_depth_image, first_depth_colormap, depth_coord)

            if second_depth_image is not None:
                second_depth_colormap = self._draw_on_depth_image(second_depth_image,
                                                                  second_depth_colormap, depth_coord)

        top_row = np.hstack((left_colored, right_colored))
        bottom_row = np.hstack((first_depth_colormap, second_depth_colormap))
        combined_view = np.vstack((cv2.resize(top_row, (1280, 480)), cv2.resize(bottom_row, (1280, 480))))

        cv2.imshow("Combined View (2x2)", combined_view)

    def _save_images(self,
                    left_gray_image: np.ndarray,
                    right_gray_image: np.ndarray,
                    first_depth_image: Optional[np.ndarray] = None,
                    second_depth_image: Optional[np.ndarray] = None
                    ) -> None:
        """
        Save the images to disk.

        args:
        left_gray_image (np.ndarray): Grayscale image of the left camera.
        right_gray_image (np.ndarray): Grayscale image of the right camera.
        first_depth_image (np.ndarray): First depth image.
        second_depth_image (np.ndarray): Second depth image.

        return:
        No return.
        """
        # File paths
        left_ir_dir = os.path.join(self.base_dir, "left_images")
        right_ir_dir = os.path.join(self.base_dir, "right_images")
        depth_dir = os.path.join(self.base_dir, "depth_images")

        # Paths for left and right images
        left_ir_path = os.path.join(left_ir_dir, f"left_image{self.image_index}.png")
        right_ir_path = os.path.join(right_ir_dir, f"right_image{self.image_index}.png")

        # Save the left and right grayscale images
        cv2.imwrite(left_ir_path, left_gray_image)
        cv2.imwrite(right_ir_path, right_gray_image)

        log_message = [
            f"Saved images - Left IR: {left_ir_path}, Right IR: {right_ir_path}"
        ]

        # Handle first depth image
        if first_depth_image is not None:
            depth_png_path_1 = os.path.join(depth_dir, f"depth_image1_{self.image_index}.png")
            depth_npy_path_1 = os.path.join(depth_dir, f"depth_image1_{self.image_index}.npy")
            cv2.imwrite(depth_png_path_1, first_depth_image)
            np.save(depth_npy_path_1, first_depth_image)

            log_message.extend([
                f"Depth PNG 1: {depth_png_path_1}",
                f"Depth NPY 1: {depth_npy_path_1}"
            ])

        # Handle second depth image
        if second_depth_image is not None:
            depth_png_path_2 = os.path.join(depth_dir, f"depth_image2_{self.image_index}.png")
            depth_npy_path_2 = os.path.join(depth_dir, f"depth_image2_{self.image_index}.npy")
            cv2.imwrite(depth_png_path_2, second_depth_image)
            np.save(depth_npy_path_2, second_depth_image)

            log_message.extend([
                f"Depth PNG 2: {depth_png_path_2}",
                f"Depth NPY 2: {depth_npy_path_2}"
            ])

        # Log all the saved paths
        logging.info(", ".join(log_message))

        # Increment image index
        self.image_index += 1
