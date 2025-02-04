"""
Main function to start the application with available camera systems.
"""

if __name__ == "__main__":
    import logging

    import pyrealsense2 as rs
    from .ui_objects import OpencvUIController
    from .camera_objects import RealsenseCameraSystem, DualRealsenseSystem, FlirCameraSystem, DualFlirSystem

    logging.disable(logging.CRITICAL)  # Suppress log messages below CRITICAL level

    def start_ui_with_camera_system(controller: OpencvUIController, camera_system, system_prefix, focal_length, baseline, principal_point):
        logging.disable(logging.NOTSET)  # Re-enable logging after tests

        controller.set_parameters(system_prefix, focal_length, baseline, principal_point)
        controller.set_camera_system(camera_system)
        controller.start()

    # Initialize UI controller without parameters
    ui_controller = OpencvUIController()

    # Try to detect dual Realsense cameras
    try:
        context = rs.context()
        connected_devices = context.query_devices()
        if len(connected_devices) >= 2:
            # D415
            FOCAL_LENGTH = 908.36  # in pixels
            BASELINE = 150  # in mm
            WIDTH = 1280
            HEIGHT = 720
            PRINCIPAL_POINT = (614.695, 354.577)  # in pixels

            camera1 = RealsenseCameraSystem(WIDTH, HEIGHT, connected_devices[0].get_info(rs.camera_info.serial_number))
            camera2 = RealsenseCameraSystem(WIDTH, HEIGHT, connected_devices[1].get_info(rs.camera_info.serial_number))
            cameras = DualRealsenseSystem(camera1, camera2)
            start_ui_with_camera_system(ui_controller, cameras, "Realsense", FOCAL_LENGTH, BASELINE, PRINCIPAL_POINT)
            exit()
    except Exception as e:
        pass

    # Try to detect single Realsense camera
    try:
        if len(connected_devices) >= 1:
            # D415
            FOCAL_LENGTH = 908.36  # in pixels
            BASELINE = 55  # in mm
            WIDTH = 1280
            HEIGHT = 720
            PRINCIPAL_POINT = (614.695, 354.577)  # in pixels

            cameras = RealsenseCameraSystem(width=WIDTH, height=HEIGHT)
            start_ui_with_camera_system(ui_controller, cameras, "Realsense", FOCAL_LENGTH, BASELINE, PRINCIPAL_POINT)
            exit()
    except Exception as e:
        pass

    # Try to detect FLIR cameras
    try:
        FOCAL_LENGTH = 1060  # in pixels
        BASELINE = 80  # in mm
        PRINCIPAL_POINT = (640, 360)  # unchecked in pixels

        CONFIG = "./src/camera_config/GH3_camera_config.yaml"
        SN1 = "21091478"
        SN2 = "21091470"

        camera1 = FlirCameraSystem(CONFIG, SN1)
        camera2 = FlirCameraSystem(CONFIG, SN2)
        cameras = DualFlirSystem(camera1, camera2)
        start_ui_with_camera_system(ui_controller, cameras, "GH3", FOCAL_LENGTH, BASELINE, PRINCIPAL_POINT)
        exit()
    except Exception as e:
        pass

    # If no cameras are detected, run the loading functionality
    ui_controller.start()
