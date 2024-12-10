import os
from datetime import datetime
import logging

import cv2
import numpy as np

from camera_objects.camera_abstract_class import Two_Cameras_System
from aruco_detector import ArUco_Detector

class OpenCV_UI_Controller():
    def __init__(self, system_prefix: str, focal_length: float, baseline: float):
        self.base_dir = os.path.join("Db", f"{system_prefix}_{datetime.now().strftime('%Y%m%d')}")
        self.left_ir_dir = os.path.join(self.base_dir, "left_images")
        self.right_ir_dir = os.path.join(self.base_dir, "right_images")
        self.depth_dir = os.path.join(self.base_dir, "depth_images")

        self._setup_directories()
        self.image_index = self._get_starting_index(self.left_ir_dir)

        self._setup_logging()

        self.mouse_x = 0
        self.mouse_y = 0
        self._setup_window()

        self.camera_system = None
        self.ArUco_detector = ArUco_Detector()

        self.focal_length = focal_length
        self.baseline = baseline

    def _setup_directories(self):
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.left_ir_dir, exist_ok=True)
        os.makedirs(self.right_ir_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

    def _get_starting_index(self, directory: str):
        if not os.path.exists(directory):
            return 1
        files = [f for f in os.listdir(directory) if f.endswith(".png")]
        indices = [
            int(os.path.splitext(f)[0].split("image")[-1])
            for f in files
        ]
        return max(indices, default=0) + 1

    def _setup_logging(self):
        log_path = os.path.join(self.base_dir, "aruco_depth_log.txt")
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def _setup_window(self):
        def _mouse_callback(event, x, y, flags, param):
            """Update the mouse position."""
            if event == cv2.EVENT_MOUSEMOVE:
                self.mouse_x, self.mouse_y = x, y

        # Create a window and set the mouse callback
        cv2.namedWindow("Combined View (2x2)")
        cv2.setMouseCallback("Combined View (2x2)", _mouse_callback)

    def set_camera_system(self, camera_system: Two_Cameras_System):
        self.camera_system = camera_system

    def _process_data(self, matching_corners_left, matching_corners_right):
        """
        處理匹配標記的視差、深度計算以及統計數據。
        
        返回視差的平均值、變異數、計算的深度以及標記中心座標。
        """
        disparities = np.abs(matching_corners_left[:, 0] - matching_corners_right[:, 0])
        mean_disparity = np.mean(disparities)
        variance_disparity = np.var(disparities)

        if mean_disparity > 0:  # 避免除以零的情況
            depth_mm_calc = (self.focal_length * self.baseline) / mean_disparity
        else:
            depth_mm_calc = 0

        # 計算標記的中心點
        center_x, center_y = np.mean(matching_corners_left, axis=0).astype(int)

        return disparities, mean_disparity, variance_disparity, depth_mm_calc, center_x, center_y

    def _draw_on_images(self, left_image_colored, right_image_colored, depth_image, matching_ids_result, matching_corners_left, matching_corners_right):
        """
        在左、右影像上繪製標記ID和計算的深度。
        """
        if depth_image is not None:
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Map the mouse position from IR to depth
            if 0 <= self.mouse_x < 640 and 0 <= self.mouse_y < 480:
                depth_x = int(self.mouse_x * (self.camera_system.get_width() / 640))
                depth_y = int(self.mouse_y * (self.camera_system.get_height() / 480))
                depth_mm_mouse = depth_image[depth_y, depth_x]

                cv2.circle(depth_colormap, (depth_x, depth_y), 5, (0, 255, 255), -1)
                text = f"Depth: {depth_mm_mouse} mm"
                cv2.putText(depth_colormap, text, (depth_x + 10, depth_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        else:
            depth_colormap = np.zeros_like(left_image_colored)

        for marker_id in matching_ids_result:
            disparities, mean_disparity, variance_disparity, depth_mm_calc, center_x, center_y = self._process_data(
                matching_corners_left, matching_corners_right)
            
            depth_mm_aruco = self._get_depth_data(depth_image, center_x, center_y)

            # 在左影像上顯示標記ID和深度
            cv2.putText(left_image_colored, f"ID:{marker_id} Depth:{int(depth_mm_calc)}mm",
                        (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 記錄資訊
            logging.info(f"Marker ID: {marker_id}, Calculated Depth: {depth_mm_calc:.2f} mm, Depth Image Depth: {depth_mm_aruco} mm, Mean Disparity: {mean_disparity:.2f}, Disparity Variance: {variance_disparity:.2f}, Disparities: {disparities.tolist()}")

        return left_image_colored, right_image_colored, depth_colormap

    def _get_depth_data(self, depth_image, center_x, center_y):
        """
        從深度影像獲取深度數據，並在深度影像上繪製。
        """
        if depth_image is not None:
            # 獲取深度值
            depth_x = min(max(int(center_x), 0), 1279)
            depth_y = min(max(int(center_y), 0), 719)
            depth_mm_img = depth_image[depth_y, depth_x]
        else:
            depth_mm_img = "Unknown"

        return depth_mm_img

    def _display_image(self, left_gray_image, right_gray_image, matching_ids_result, matching_corners_left, matching_corners_right, depth_image=None):
        """
        顯示左右影像和深度影像，並繪製標記ID和深度。
        """
        # 將灰階影像轉換為彩色影像以便顯示
        left_image_colored = cv2.cvtColor(left_gray_image, cv2.COLOR_GRAY2BGR)
        right_image_colored = cv2.cvtColor(right_gray_image, cv2.COLOR_GRAY2BGR)

        # 在左、右影像上繪製標記ID和深度
        left_image_colored, right_image_colored, depth_colormap = self._draw_on_images(left_image_colored, right_image_colored, depth_image, matching_ids_result, matching_corners_left, matching_corners_right) 

        # 創建 2x2 視圖矩陣
        top_row = np.hstack((left_image_colored, right_image_colored))  # 合併左右影像
        top_row_resized = cv2.resize(top_row, (1280, 480))  # 調整大小以符合下方矩陣寬度
        bottom_row = np.hstack((depth_colormap, np.zeros_like(depth_colormap)))  # 合併深度影像和佔位圖
        bottom_row_resized = cv2.resize(bottom_row, (1280, 480))  # 調整大小以符合下方矩陣寬度
        combined_view = np.vstack((top_row_resized, bottom_row_resized))  # 垂直堆疊形成最終視圖

        # 顯示最終影像
        cv2.imshow("Combined View (2x2)", combined_view)

    def _save_images(self, left_gray_image, right_gray_image, depth_image = None):
        # File paths
        left_ir_path = os.path.join(self.left_ir_dir, f"left_image{self.image_index}.png")
        right_ir_path = os.path.join(self.right_ir_dir, f"right_image{self.image_index}.png")
        
        # Save images
        cv2.imwrite(left_ir_path, left_gray_image)
        cv2.imwrite(right_ir_path, right_gray_image)
        
        if depth_image is not None:
            depth_png_path = os.path.join(self.depth_dir, f"depth_image{self.image_index}.png")
            depth_npy_path = os.path.join(self.depth_dir, f"depth_image{self.image_index}.npy")
            cv2.imwrite(depth_png_path, depth_image)
            np.save(depth_npy_path, depth_image)

            logging.info(f"Saved images - Left IR: {left_ir_path}, Right IR: {right_ir_path}, Depth PNG: {depth_png_path}, Depth NPY: {depth_npy_path}")
        else:
            logging.info(f"Saved images - Left IR: {left_ir_path}, Right IR: {right_ir_path}")

        self.image_index += 1

    def start(self):
        while True:
            success, left_gray_image, right_gray_image = self.camera_system.get_grayscale_images()
            depth_success, depth_image = self.camera_system.get_depth_image()
            if not success:
                continue
            else:
                matching_ids_result, matching_corners_left, matching_corners_right = self.ArUco_detector.detect_aruco_two_images(left_gray_image, right_gray_image)
                self._display_image(left_gray_image, right_gray_image, matching_ids_result, matching_corners_left, matching_corners_right, depth_image)

            # Check for key presses
            key = cv2.pollKey() & 0xFF
            if key == 27:  # ESC key
                logging.info("Program terminated by user.")
                self.camera_system.release()
                cv2.destroyAllWindows()
                break
            elif key == ord('s') or key == ord('S'):  # Save images
                self._save_images(left_gray_image, right_gray_image, depth_image)
