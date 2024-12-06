import pyrealsense2 as rs
import numpy as np
import cv2
import os
import logging
from datetime import datetime

# Logging configuration
base_dir = os.path.join("Db", f"Realsense_{datetime.now().strftime('%Y%m%d')}")
log_path = os.path.join(base_dir, "aruco_depth_log.txt")
os.makedirs(base_dir, exist_ok=True)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Program started")

# Define paths for saving images
left_ir_dir = os.path.join(base_dir, "left_IR_images")
right_ir_dir = os.path.join(base_dir, "right_IR_images")
depth_dir = os.path.join(base_dir, "depth_images")

# Ensure directories exist
os.makedirs(left_ir_dir, exist_ok=True)
os.makedirs(right_ir_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

# Determine the starting index for image saving
def get_starting_index(directory):
    if not os.path.exists(directory):
        return 1
    files = [f for f in os.listdir(directory) if f.endswith(".png")]
    indices = [
        int(os.path.splitext(f)[0].split("image")[-1])
        for f in files if f.startswith("depth_image")
    ]
    return max(indices, default=0) + 1

image_index = get_starting_index(depth_dir)

# Configure the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable the depth and infrared streams at 1280x720 resolution
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # Depth
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)  # Left IR (Y8)
config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)  # Right IR (Y8)

# Start the pipeline
pipeline.start(config)

# Camera intrinsic parameters
focal_length = 908.36  # in pixels
baseline = 55  # in mm

# Define ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()

# Mouse interaction variables
mouse_x, mouse_y = -1, -1

def mouse_callback(event, x, y, flags, param):
    """Update the mouse position."""
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

# Create a window and set the mouse callback
cv2.namedWindow("Combined View (2x2)")
cv2.setMouseCallback("Combined View (2x2)", mouse_callback)

try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        ir_frame_left = frames.get_infrared_frame(1)  # Left IR
        ir_frame_right = frames.get_infrared_frame(2)  # Right IR

        if not depth_frame or not ir_frame_left or not ir_frame_right:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        ir_image_left = np.asanyarray(ir_frame_left.get_data())
        ir_image_right = np.asanyarray(ir_frame_right.get_data())

        # Convert grayscale infrared images to 3 channels for visualization
        ir_left_colored = cv2.cvtColor(ir_image_left, cv2.COLOR_GRAY2BGR)
        ir_right_colored = cv2.cvtColor(ir_image_right, cv2.COLOR_GRAY2BGR)

        # Detect ArUco markers in both original IR images
        corners_left, ids_left, _ = cv2.aruco.detectMarkers(ir_image_left, aruco_dict, parameters=parameters)
        corners_right, ids_right, _ = cv2.aruco.detectMarkers(ir_image_right, aruco_dict, parameters=parameters)

        # Match ArUco markers by ID and calculate disparity and depth
        if ids_left is not None and ids_right is not None:
            ids_left = ids_left.flatten()
            ids_right = ids_right.flatten()
            matching_ids = set(ids_left).intersection(ids_right)

            for marker_id in matching_ids:
                idx_left = np.where(ids_left == marker_id)[0][0]
                idx_right = np.where(ids_right == marker_id)[0][0]

                # Get corners for the matching marker
                corners_l = corners_left[idx_left][0]
                corners_r = corners_right[idx_right][0]

                # Calculate disparity for the 4 corners
                disparities = np.abs(corners_l[:, 0] - corners_r[:, 0])
                mean_disparity = np.mean(disparities)
                variance_disparity = np.var(disparities)

                if mean_disparity > 0:  # Avoid division by zero
                    depth_mm_calc = (focal_length * baseline) / mean_disparity
                else:
                    depth_mm_calc = 0

                # Calculate marker center
                center_x, center_y = np.mean(corners_l, axis=0).astype(int)

                # Get depth value from the depth image at the marker center
                depth_x = min(max(int(center_x), 0), 1279)
                depth_y = min(max(int(center_y), 0), 719)
                depth_mm_img = depth_image[depth_y, depth_x]

                # Display ArUco ID and depth at the marker center
                cv2.putText(ir_left_colored, f"ID:{marker_id} Depth:{int(depth_mm_calc)}mm",
                            (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Log the data
                logging.info(f"Marker ID: {marker_id}, Calculated Depth: {depth_mm_calc:.2f} mm, Depth Image Depth: {depth_mm_img} mm, Mean Disparity: {mean_disparity:.2f}, Disparity Variance: {variance_disparity:.2f}, Disparities: {disparities.tolist()}")

        # Resize depth image for display
        depth_resized = cv2.resize(depth_image, (640, 480), interpolation=cv2.INTER_AREA)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_resized, alpha=0.03), cv2.COLORMAP_JET)

        # Map the mouse position from IR to depth
        if 0 <= mouse_x < 640 and 0 <= mouse_y < 480:
            depth_x = int(mouse_x * (1280 / 640))
            depth_y = int(mouse_y * (720 / 480))
            depth_value = depth_image[depth_y, depth_x]

            # Overlay cursor and depth value on the depth image
            cv2.circle(depth_colormap, (mouse_x, mouse_y), 5, (0, 255, 255), -1)
            text = f"Depth: {depth_value} mm"
            cv2.putText(depth_colormap, text, (mouse_x + 10, mouse_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Create a 2x2 matrix view
        top_row = np.hstack((ir_left_colored, ir_right_colored))  # Combine left and right IR images
        top_row_resized = cv2.resize(top_row, (1280, 480))  # Resize to match the width of the bottom row
        bottom_row = np.hstack((depth_colormap, np.zeros_like(depth_colormap)))  # Combine depth image and placeholder
        combined_view = np.vstack((top_row_resized, bottom_row))  # Stack rows to create the combined view

        # Show the combined image
        cv2.imshow("Combined View (2x2)", combined_view)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            logging.info("Program terminated by user.")
            break
        elif key == ord('s') or key == ord('S'):  # Save images
            # File paths
            left_ir_path = os.path.join(left_ir_dir, f"left_IR_image{image_index}.png")
            right_ir_path = os.path.join(right_ir_dir, f"right_IR_image{image_index}.png")
            depth_png_path = os.path.join(depth_dir, f"depth_image{image_index}.png")
            depth_npy_path = os.path.join(depth_dir, f"depth_image{image_index}.npy")

            # Save images
            cv2.imwrite(left_ir_path, ir_image_left)
            cv2.imwrite(right_ir_path, ir_image_right)
            cv2.imwrite(depth_png_path, depth_image)
            np.save(depth_npy_path, depth_image)

            logging.info(f"Saved images - Left IR: {left_ir_path}, Right IR: {right_ir_path}, Depth PNG: {depth_png_path}, Depth NPY: {depth_npy_path}")
            image_index += 1

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
