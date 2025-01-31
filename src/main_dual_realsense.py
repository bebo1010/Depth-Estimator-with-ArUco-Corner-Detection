"""
Main function to start the application with Realsense camera system.
"""

if __name__ == "__main__":
    import pyrealsense2 as rs

    from .ui_objects import OpencvUIController
    from .camera_objects import RealsenseCameraSystem, DualRealsenseSystem
    # D415
    FOCAL_LENGTH = 908.36  # in pixels
    BASELINE = 150  # in mm
    WIDTH = 1280
    HEIGHT = 720
    PRINCIPAL_POINT = (614.695, 354.577)  # in pixels

    # D435
    # FOCAL_LENGTH = 425.203  # in pixels
    # BASELINE = 50  # in mm
    # WIDTH = 848
    # HEIGHT = 480
    # PRINCIPAL_POINT = (424, 240)  # in pixels

    UI = OpencvUIController(system_prefix="Realsense",
                            focal_length=FOCAL_LENGTH,
                            baseline=BASELINE,
                            principal_point=PRINCIPAL_POINT)

    # Create a context object. This object manages all connected devices
    context = rs.context()

    # Get a list of all connected devices
    connected_devices = context.query_devices()

    if len(connected_devices) < 2:
        raise ValueError(f"Not enough cameras, only detected {len(connected_devices)} cameras.")

    # Initialize the two RealsenseCameraSystem instances
    camera1 = RealsenseCameraSystem(WIDTH, HEIGHT, connected_devices[0].get_info(rs.camera_info.serial_number))
    camera2 = RealsenseCameraSystem(WIDTH, HEIGHT, connected_devices[1].get_info(rs.camera_info.serial_number))

    cameras = DualRealsenseSystem(camera1, camera2)
    UI.set_camera_system(cameras)

    UI.start()
