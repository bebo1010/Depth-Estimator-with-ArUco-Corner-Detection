"""
Module for calibrating cameras using a chessboard pattern.
"""

import os
from datetime import datetime
import logging
from typing import Tuple, List, Dict, Any

import json

import cv2
import numpy as np

class ChessboardCalibrator():
    """
    Class for calibrating single and stereo cameras using a chessboard pattern.
    """
    def __init__(self,
                 pattern_size: Tuple[int, int] = (11, 7),
                 square_size_mm: float = 30.0
                 ) -> None:
        self._pattern_size = pattern_size
        self._square_size_mm = square_size_mm

        self._left_calibration_parameters: Dict[str, Any] = {}
        self._right_calibration_parameters: Dict[str, Any] = {}
        self._stereo_calibration_parameters: Dict[str, Any] = {}

        self.rectify_undistort_maps: Dict[str, Any] = {
            "mapx_left": None,
            "mapy_left": None,
            "mapx_right": None,
            "mapy_right": None
        }
        logging.info("ChessboardCalibrator initialized with pattern size %s and square size mm %s.",
                        pattern_size, square_size_mm)

    @property
    def pattern_size(self) -> Tuple[int, int]:
        """
        Get the pattern size of the chessboard.

        Returns
        -------
        Tuple[int, int]
            Pattern size of the chessboard.
        """
        return self._pattern_size

    @pattern_size.setter
    def pattern_size(self, new_pattern_size: Tuple[int, int]) -> None:
        """
        Set a new pattern size for the chessboard.

        Parameters
        ----------
        new_pattern_size : Tuple[int, int]
            New pattern size of the chessboard.
        """
        self._pattern_size = new_pattern_size
        logging.info("Pattern size set to %s.", new_pattern_size)

    @property
    def square_size_mm(self) -> float:
        """
        Get the square size of the chessboard in millimeters.

        Returns
        -------
        float
            Square size of the chessboard in millimeters.
        """
        return self._square_size_mm

    @square_size_mm.setter
    def square_size_mm(self, new_square_size_mm: float) -> None:
        """
        Set a new square size for the chessboard in millimeters.

        Parameters
        ----------
        new_square_size_mm : float
            New square size of the chessboard in millimeters.
        """
        self._square_size_mm = new_square_size_mm
        logging.info("Square size mm set to %f.", new_square_size_mm)

    @property
    def left_camera_parameters(self) -> Dict[str, Any]:
        """
        Get the calibration parameters for the left camera.

        Returns
        -------
        Dict[str, Any]
            Calibration parameters for the left camera.
        """
        return self._left_calibration_parameters

    @left_camera_parameters.setter
    def left_camera_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set the calibration parameters for the left camera.

        Parameters
        ----------
        parameters : Dict[str, Any]
            Calibration parameters for the left camera.
        """
        self._left_calibration_parameters = parameters
        logging.info("Left camera parameters set.")

    @property
    def right_camera_parameters(self) -> Dict[str, Any]:
        """
        Get the calibration parameters for the right camera.

        Returns
        -------
        Dict[str, Any]
            Calibration parameters for the right camera.
        """
        return self._right_calibration_parameters

    @right_camera_parameters.setter
    def right_camera_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set the calibration parameters for the right camera.

        Parameters
        ----------
        parameters : Dict[str, Any]
            Calibration parameters for the right camera.
        """
        self._right_calibration_parameters = parameters
        logging.info("Right camera parameters set.")

    @property
    def stereo_camera_parameters(self) -> Dict[str, Any]:
        """
        Get the calibration parameters for the stereo camera setup.

        Returns
        -------
        Dict[str, Any]
            Calibration parameters for the stereo camera setup.
        """
        return self._stereo_calibration_parameters

    @stereo_camera_parameters.setter
    def stereo_camera_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set the calibration parameters for the stereo camera setup.

        Parameters
        ----------
        parameters : Dict[str, Any]
            Calibration parameters for the stereo camera setup.
        """
        self._stereo_calibration_parameters = parameters
        logging.info("Stereo camera parameters set.")

    def detect_chessboard_corners(self, image: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Detect chessboard corners in a single image, and return detected corners and IDs.

        Parameters
        ----------
        image : np.ndarray
            Image, should include chessboard pattern.

        Returns
        -------
        Tuple[bool, np.ndarray]
            - bool: Whether corners were detected.
            - np.ndarray: Detected corners in image.
        """
        logging.info("Detecting chessboard corners in image.")
        ret, corners = cv2.findChessboardCorners(image, self.pattern_size)

        if ret:
            logging.info("Chessboard corners detected, running subpixel refinement.")
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(image, corners, (5, 5), (-1, -1), criteria)
            logging.info("Chessboard corners refined.")
            return ret, corners

        logging.warning("Chessboard corners not detected.")
        return ret, None

    def display_chessboard_corners(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Display chessboard corners on an image.

        Parameters
        ----------
        image : np.ndarray
            Image, should include chessboard pattern.
        corners : np.ndarray
            Detected corners in image.

        Returns
        -------
        np.ndarray
            Image with corners drawn.
        """
        logging.info("Drawing chessboard corners on image.")
        return cv2.drawChessboardCorners(image, self.pattern_size, corners, True)

    def calibrate_single_camera(self,
                         image_points: List[np.ndarray],
                         image_size: Tuple[int, int],
                         camera_index: int = 0
                         ) -> bool:
        """
        Calibrate a single camera using detected image points.

        Parameters
        ----------
        image_points : List[np.ndarray]
            List of detected image points.
        image_size : Tuple[int, int]
            Size of the image.
        camera_index : int, optional
            Index of the camera. `0` for left camera, `1` for right camera.

        Returns
        -------
        bool
            Whether calibration was successful.
        """
        assert len(image_points) > 0, "No image points detected."
        obj_points = self._generate_object_points(len(image_points))
        logging.info("Calibrating camera with %d images for camera %d", len(image_points), camera_index)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, image_points, image_size,
                                                           cameraMatrix=None, distCoeffs=None)
        logging.info("Camera calibration for camera %d complete with mean reprojection error: %f.", camera_index,  ret)

        # arbitrary threshold for successful calibration
        if ret > 3.0:
            logging.error("Camera calibration failed.")
            return False

        if camera_index == 0:
            self._left_calibration_parameters = {
                "image_points": image_points,
                "object_points": obj_points,
                "camera_matrix": mtx,
                "distortion_coefficients": dist,
                "rotation_vectors": rvecs,
                "translation_vectors": tvecs
            }
        elif camera_index == 1:
            self._right_calibration_parameters = {
                "image_points": image_points,
                "object_points": obj_points,
                "camera_matrix": mtx,
                "distortion_coefficients": dist,
                "rotation_vectors": rvecs,
                "translation_vectors": tvecs
            }

        return True

    def calibrate_stereo_camera(self,
                                left_image_points: List[np.ndarray],
                                right_image_points: List[np.ndarray],
                                image_size: Tuple[int, int]
                                ) -> bool:
        """
        Calibrate a stereo camera setup using detected image points from both cameras.

        Parameters
        ----------
        left_image_points : List[np.ndarray]
            List of detected image points from the left camera.
        right_image_points : List[np.ndarray]
            List of detected image points from the right camera.
        image_size : Tuple[int, int]
            Size of the image.

        Returns
        -------
        bool
            Whether calibration was successful.
        """
        assert len(left_image_points) == len(right_image_points), \
             "Number of points in left and right images must be equal."

        single_calibration_success = True
        single_calibration_success &= self.calibrate_single_camera(left_image_points, image_size, camera_index=0)
        single_calibration_success &= self.calibrate_single_camera(right_image_points, image_size, camera_index=1)
        if not single_calibration_success:
            logging.error("Stereo camera calibration failed.")
            return False

        logging.info("Single camera calibration successful for both cameras.")

        mtx_left = self._left_calibration_parameters["camera_matrix"]
        dist_left = self._left_calibration_parameters["distortion_coefficients"]
        mtx_right = self._right_calibration_parameters["camera_matrix"]
        dist_right = self._right_calibration_parameters["distortion_coefficients"]

        obj_points = self._generate_object_points(len(left_image_points))
        ret, new_mtx_left, new_dist_left, new_mtx_right, new_dist_right, \
            rotation, translation, essential, fundamental = cv2.stereoCalibrate(
            obj_points, left_image_points, right_image_points,
            mtx_left, dist_left, mtx_right, dist_right,
            image_size, flags=None, criteria=None
            )
        logging.info("Stereo camera calibration complete with mean reprojection error: %f.", ret)

        # arbitrary threshold for successful calibration
        if ret > 3.0:
            logging.error("Stereo camera calibration failed.")
            return False

        logging.info("Rectifying stereo camera setup.")
        left_rectified_rotation, right_rectified_rotation, \
              left_projection, right_projection, \
                reprojection, \
                    left_roi, right_roi = \
                        cv2.stereoRectify(new_mtx_left, new_dist_left, new_mtx_right, new_dist_right, \
                                          image_size, rotation, translation)
        logging.info("Stereo camera rectification complete.")

        self._stereo_calibration_parameters = {
            "image_points_left": left_image_points,
            "image_points_right": right_image_points,
            "object_points": obj_points,
            "camera_matrix_left": new_mtx_left,
            "distortion_coefficients_left": new_dist_left,
            "camera_matrix_right": new_mtx_right,
            "distortion_coefficients_right": new_dist_right,
            "rotation_matrix": rotation,
            "translation_vector": translation,
            "essential_matrix": essential,
            "fundamental_matrix": fundamental,
            "left_rectified_rotation_matrix": left_rectified_rotation,
            "right_rectified_rotation_matrix": right_rectified_rotation,
            "left_projection_matrix": left_projection,
            "right_projection_matrix": right_projection,
            "reprojection_matrix": reprojection,
            "left_rectified_roi": left_roi,
            "right_rectified_roi": right_roi
        }

        return True

    def rectify_images(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify a pair of stereo images using the calibration parameters.

        Parameters
        ----------
        left_image : np.ndarray
            Image from the left camera.
        right_image : np.ndarray
            Image from the right camera.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Rectified images from the left and right cameras.
        """
        logging.info("Rectifying stereo images.")
        if not np.all([map is not None for map in self.rectify_undistort_maps.values()]):
            self._initialize_rectification_maps((left_image.shape[1], left_image.shape[0]))

        left_map1 = self.rectify_undistort_maps["mapx_left"]
        left_map2 = self.rectify_undistort_maps["mapy_left"]

        right_map1 = self.rectify_undistort_maps["mapx_right"]
        right_map2 = self.rectify_undistort_maps["mapy_right"]

        left_rectified = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)

        logging.info("Stereo images rectified.")
        return left_rectified, right_rectified

    def save_parameters(self, db_path: str = "./", system_prefix = "GH3") -> None:

        """
        Save the calibration parameters to JSON files.

        Parameters
        ----------
        db_path : str, optional
            Directory path to save the JSON files. Default is the current directory.

        system_prefix : str, optional
            Prefix for the directory name where the JSON files will be saved. Default is "GH3".

        Raises
        ------
        TypeError
            If an object in the calibration parameters is not JSON serializable.
        """
        dir_path = os.path.join(db_path, f"{system_prefix}_calibration_parameter_{datetime.now().strftime('%Y%m%d')}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        left_parameter_save_path = os.path.join(dir_path, "left_camera_parameters.json")
        right_parameter_save_path = os.path.join(dir_path, "right_camera_parameters.json")
        stereo_parameter_save_path = os.path.join(dir_path, "stereo_camera_parameters.json")

        def serialize_numpy(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(left_parameter_save_path, "w", encoding="utf-8") as left_file:
            json.dump(self._left_calibration_parameters, left_file, default=serialize_numpy)

        with open(right_parameter_save_path, "w", encoding="utf-8") as right_file:
            json.dump(self._right_calibration_parameters, right_file, default=serialize_numpy)

        with open(stereo_parameter_save_path, "w", encoding="utf-8") as stereo_file:
            json.dump(self._stereo_calibration_parameters, stereo_file, default=serialize_numpy)

    def _generate_object_points(self, num_images: int) -> List[np.ndarray]:
        """
        Generate object points for the chessboard pattern.

        Parameters
        ----------
        num_images : int
            Number of images.

        Returns
        -------
        List[np.ndarray]
            List of object points.
        """
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        return [objp for _ in range(num_images)]

    def _initialize_rectification_maps(self, image_size: Tuple[int, int]) -> None:
        """
        Initialize rectification maps for the stereo camera setup.

        Parameters
        ----------
        image_size : Tuple[int, int]
            Size of the image.
        """
        # Note: Currently untested
        if len(self._stereo_calibration_parameters.keys()) > 0:
            left_map1, left_map2 = cv2.initUndistortRectifyMap(
                self._stereo_calibration_parameters["camera_matrix_left"],
                self._stereo_calibration_parameters["distortion_coefficients_left"],
                self._stereo_calibration_parameters["left_rectified_rotation_matrix"],
                self._stereo_calibration_parameters["left_projection_matrix"],
                image_size,
                cv2.CV_32FC1
            )

            right_map1, right_map2 = cv2.initUndistortRectifyMap(
                self._stereo_calibration_parameters["camera_matrix_right"],
                self._stereo_calibration_parameters["distortion_coefficients_right"],
                self._stereo_calibration_parameters["right_rectified_rotation_matrix"],
                self._stereo_calibration_parameters["right_projection_matrix"],
                image_size,
                cv2.CV_32FC1
            )

            self.rectify_undistort_maps = {
                "mapx_left": left_map1,
                "mapy_left": left_map2,
                "mapx_right": right_map1,
                "mapy_right": right_map2
            }
            logging.info("Rectification maps initialized.")
            return

        logging.error("Stereo camera parameters not set. Cannot initialize rectification maps.")
