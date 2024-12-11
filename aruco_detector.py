"""
Module for detecting ArUco markers in images.
"""
from typing import Tuple

import cv2
import numpy as np

class ArUcoDetector():
    """
    Detect ArUco markers in images

    Functions:
        __init__() -> None
        detect_aruco(np.ndarray) -> Tuple[np.ndarray, np.ndarray]
        detect_aruco_two_images(np.ndarray, np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]

    """
    def __init__(self) -> None:
        """
        Initialize detector for ArUco markers.

        args:
        No arguments
        returns:
        No returns.
        """
        # Define ArUco dictionary and parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.parameters = cv2.aruco.DetectorParameters()

    def detect_aruco(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect ArUco markers in a single image, and return detected IDs and corner points.
        
        args:
        image (np.ndarray): Image, should include ArUco markers in image.

        returns:
        Tuple[np.ndarray, np.ndarray]: 
            - np.ndarray: Detected IDs in image.
            - np.ndarray: Detected corner points in image.
        """
        corners, ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.parameters)
        return corners, ids

    def detect_aruco_two_images(self,
                                image_left: np.ndarray,
                                image_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect ArUco markers in two images, and return matching IDs and corner points.
        
        args:
        image_left (np.ndarray): Left image, should include ArUco markers in image.
        image_right (np.ndarray): Right image, should include ArUco markers in image.

        returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - np.ndarray: Matching IDs in both image.
            - np.ndarray: Matching corner points in left image.
            - np.ndarray: Matching corner points in right image.
        """
        # 偵測左影像中的Aruco標記，並提取角落點與ID
        corners_left, ids_left = self.detect_aruco(image_left)

        # 偵測右影像中的Aruco標記，並提取角落點與ID
        corners_right, ids_right = self.detect_aruco(image_right)

        # 檢查兩張影像是否都有偵測到標記
        if ids_left is not None and ids_right is not None:
            # 將ID列表轉換為一維陣列
            ids_left = ids_left.flatten()
            ids_right = ids_right.flatten()

            # 找出兩張影像中匹配的ID
            matching_ids = set(ids_left).intersection(ids_right)

            # 定義用來儲存匹配標記的角落點和ID
            matching_corners_left = []
            matching_corners_right = []
            matching_ids_result = []

            # 對每一個匹配的ID，提取其對應的角落點
            for marker_id in matching_ids:
                # 找到左影像中匹配ID的索引
                idx_left = np.where(ids_left == marker_id)[0][0]
                # 找到右影像中匹配ID的索引
                idx_right = np.where(ids_right == marker_id)[0][0]

                # 取得匹配標記的角落點
                # 將匹配的角落點和ID儲存
                matching_corners_left.append(corners_left[idx_left][0]) # 左影像的角落點
                matching_corners_right.append(corners_right[idx_right][0]) # 右影像的角落點
                matching_ids_result.append(marker_id)

            # 返回匹配的標記ID及其角落點（左影像角落點、右影像角落點）
            matching_ids_result = np.array(matching_ids_result)
            matching_corners_left = np.array(matching_corners_left)
            matching_corners_right = np.array(matching_corners_right)

            return matching_ids_result, matching_corners_left, matching_corners_right
        # 如果其中一張影像沒有偵測到標記，返回空列表
        return np.array([]), np.array([]), np.array([])
