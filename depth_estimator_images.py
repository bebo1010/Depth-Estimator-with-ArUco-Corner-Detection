import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog

# Initialize the QApplication
app = QApplication([])

# File paths for images
left_ir_images = []
right_ir_images = []
depth_images = []

# Current image set index
current_index = 0

# Mouse interaction variables
mouse_x, mouse_y = -1, -1

# ArUco detection parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_params = cv2.aruco.DetectorParameters()

def mouse_callback(event, x, y, flags, param):
    """Update the mouse position."""
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

# Create a window and set the mouse callback
cv2.namedWindow("Combined View (2x2)")
cv2.setMouseCallback("Combined View (2x2)", mouse_callback)

def load_images_from_directory(directory):
    """Load left, right, and depth images from the selected directory."""
    global left_ir_images, right_ir_images, depth_images

    # Get all image files in the directory
    left_ir_images = sorted([os.path.join(directory, "left_IR_images", f) for f in os.listdir(os.path.join(directory, "left_IR_images")) if f.endswith('.png')])
    right_ir_images = sorted([os.path.join(directory, "right_IR_images", f) for f in os.listdir(os.path.join(directory, "right_IR_images")) if f.endswith('.png')])
    depth_images = sorted([os.path.join(directory, "depth_images", f) for f in os.listdir(os.path.join(directory, "depth_images")) if f.endswith('.png')])

    # Log the loaded images
    print(f"Loaded {len(left_ir_images)} sets of images from {directory}")

def load_image_set(index):
    """Load the left, right, and depth images for the given index."""
    left_image = cv2.imread(left_ir_images[index], cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_ir_images[index], cv2.IMREAD_GRAYSCALE)
    depth_image = cv2.imread(depth_images[index], cv2.IMREAD_UNCHANGED)  # Load depth image as is (16-bit)
    return left_image, right_image, depth_image

# Function to handle directory selection via QFileDialog
def select_directory():
    """Prompt user to select a directory for Realsense data."""
    directory = QFileDialog.getExistingDirectory(None, "Select Realsense Data Directory")
    if directory:
        load_images_from_directory(directory)

# Prompt user to select the directory when the program starts
select_directory()

# Start with an empty matrix view (blank window)
empty_image = np.zeros((480, 640, 3), dtype=np.uint8)

def detect_aruco_markers(image):
    """Detect ArUco markers in the given image."""
    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
    return corners, ids

try:
    while True:
        # If no images are loaded, show the empty matrix
        if len(left_ir_images) == 0 and len(right_ir_images) == 0 and len(depth_images) == 0:
            cv2.imshow("Combined View (2x2)", empty_image)

            print(f"No image sets were loaded!")
            # Prompt user to select the directory when the program starts
            select_directory()
            continue

        # Load the current set of images
        left_image, right_image, depth_image = load_image_set(current_index)

        # Detect ArUco markers in the left and right images
        corners_left, ids_left = detect_aruco_markers(left_image)
        corners_right, ids_right = detect_aruco_markers(right_image)

        # Convert grayscale infrared images to 3 channels for visualization
        ir_left_colored = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
        ir_right_colored = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)

        # Calculate disparity for the detected markers
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
                    depth_mm_calc = (908.36 * 55) / mean_disparity  # Using the intrinsic values
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

        # Update window title with the left image filename
        window_title = f"Combined View (2x2) -- {os.path.basename(left_ir_images[current_index])}"
        cv2.setWindowTitle("Combined View (2x2)", window_title)
        # Show the combined image
        cv2.imshow("Combined View (2x2)", combined_view)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('l') or key == ord('L'):  # Load a directory
            select_directory()
        elif key == ord('p') or key == ord('P'):  # Show previous set of images
            current_index = (current_index - 1) % len(left_ir_images)
        elif key == ord('n') or key == ord('N'):  # Show next set of images
            current_index = (current_index + 1) % len(left_ir_images)

finally:
    # Close all OpenCV windows
    cv2.destroyAllWindows()
