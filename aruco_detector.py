from typing import Tuple, List

import cv2
import numpy as np

class ArUco_Detector():
    def __init__(self):
        # Define ArUco dictionary and parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.parameters = cv2.aruco.DetectorParameters()

    def detect_aruco(self, image) -> Tuple[List[np.ndarray], List[int]]:
        corners, ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.parameters)
        return [corners, ids]
    
    def detect_aruco_two_images(self, image_left, image_right) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        偵測兩張影像中的Aruco標記，並返回兩張影像中ID相符的標記及其對應的角落點。

        這個函數會對兩張影像進行Aruco標記偵測，並比較它們的標記ID。若兩張影像有相同的標記ID，則返回這些匹配的標記及其角落點。

        參數：
        ir_image_left (np.ndarray): 第一張影像（左影像），應該包含Aruco標記。
        ir_image_right (np.ndarray): 第二張影像（右影像），應該包含Aruco標記。

        回傳：
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - np.ndarray: 兩張影像中匹配標記的ID
            - np.ndarray: 左邊影像中匹配標記的角落點
            - np.ndarray: 右邊影像中匹配標記的角落點
        """
        # 偵測左影像中的Aruco標記，並提取角落點與ID
        corners_left, ids_left, _ = cv2.aruco.detectMarkers(image_left, self.aruco_dict, parameters=self.parameters)

        # 偵測右影像中的Aruco標記，並提取角落點與ID
        corners_right, ids_right, _ = cv2.aruco.detectMarkers(image_right, self.aruco_dict, parameters=self.parameters)

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
                corners_l = corners_left[idx_left][0]  # 左影像的角落點
                corners_r = corners_right[idx_right][0]  # 右影像的角落點

                # 將匹配的角落點和ID儲存
                matching_corners_left.append(corners_l)
                matching_corners_right.append(corners_r)
                matching_ids_result.append(marker_id)

            # 返回匹配的標記ID及其角落點（左影像角落點、右影像角落點）
            matching_ids_result = np.array(matching_ids_result)
            matching_corners_left = np.squeeze(np.array(matching_corners_left))
            matching_corners_right = np.squeeze(np.array(matching_corners_right))

            return matching_ids_result, matching_corners_left, matching_corners_right
        
        else:
            # 如果其中一張影像沒有偵測到標記，返回空列表
            return np.array([]), np.array([]), np.array([])
