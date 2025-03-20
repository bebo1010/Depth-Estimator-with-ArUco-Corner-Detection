"""
Main function to start the application with available camera systems.
"""

if __name__ == "__main__":
    import logging
    import sys  # Add import for sys module

    from .ui_objects import OpencvUIController

    logging.disable(logging.CRITICAL)  # Suppress log messages below CRITICAL level

    def start_ui_with_camera_system(controller: OpencvUIController,
                                    camera_system, system_prefix,
                                    focal_length, baseline,
                                    principal_point):
        """
        Start the UI with the specified camera system and parameters.

        Args:
            controller (OpencvUIController): The UI controller to manage the UI.
            camera_system: The camera system to be used.
            system_prefix (str): The prefix for the camera system.
            focal_length (float): The focal length of the camera system in pixels.
            baseline (float): The baseline distance between cameras in mm.
            principal_point (tuple): The principal point of the camera system in pixels.
        """
        logging.disable(logging.NOTSET)  # Re-enable logging after tests

        controller.set_parameters(system_prefix, focal_length, baseline, principal_point)
        controller.set_camera_system(camera_system)
        controller.start()

    # Initialize UI controller without parameters
    ui_controller = OpencvUIController()

    try:
        # try to import realsense SDK
        import pyrealsense2 as rs
        from .camera_objects import RealsenseCameraSystem, DualRealsenseSystem

        # Try to detect dual Realsense cameras
        try:
            context = rs.context()
            connected_devices = context.query_devices()
            if len(connected_devices) >= 2:
                # # D415
                FOCAL_LENGTH = 908.36  # in pixels
                BASELINE = 150  # in mm
                WIDTH = 1280
                HEIGHT = 720
                PRINCIPAL_POINT = (614.695, 354.577)  # in pixels

                camera1 = RealsenseCameraSystem(WIDTH, HEIGHT,
                                                connected_devices[0].get_info(rs.camera_info.serial_number))
                camera2 = RealsenseCameraSystem(WIDTH, HEIGHT,
                                                connected_devices[1].get_info(rs.camera_info.serial_number))
                cameras = DualRealsenseSystem(camera1, camera2)
                start_ui_with_camera_system(ui_controller, cameras, "Dual_Realsense_D415",
                                            FOCAL_LENGTH, BASELINE, PRINCIPAL_POINT)
                sys.exit()  # Use sys.exit() instead of exit()
        except SystemExit:
            pass
        except IndexError as e:
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

                # D435
                # FOCAL_LENGTH = 425.57  # in pixels
                # BASELINE = 50  # in mm
                # WIDTH = 848
                # HEIGHT = 480
                # PRINCIPAL_POINT = (428.448, 229.037)  # in pixels

                cameras = RealsenseCameraSystem(width=WIDTH, height=HEIGHT)
                start_ui_with_camera_system(ui_controller, cameras, "Single_Realsense_D415",
                                            FOCAL_LENGTH, BASELINE, PRINCIPAL_POINT)
                sys.exit()  # Use sys.exit() instead of exit()
        except SystemExit:
            pass
        except IndexError as e:
            pass

    except ModuleNotFoundError:
        print("Realsense SDK not installed. Skipping Realsense cameras.")
        print("Run pip install pyrealsense2==2.55.1.6486 to install Realsense SDK.")

    # Try to detect FLIR cameras
    try:
        import PySpin  # pylint: disable=unused-import
        from .camera_objects import FlirCameraSystem, DualFlirSystem

        FOCAL_LENGTH = 1072  # in pixels
        BASELINE = 80  # in mm
        PRINCIPAL_POINT = (945.68, 548.12)  # unchecked in pixels

        CONFIG = "./src/camera_config/GH3_camera_config.yaml"
        SN1 = "21091478"
        SN2 = "21091470"

        camera1 = FlirCameraSystem(CONFIG, SN1)
        camera2 = FlirCameraSystem(CONFIG, SN2)
        cameras = DualFlirSystem(camera1, camera2, synchronized=True)
        start_ui_with_camera_system(ui_controller, cameras, "GH3", FOCAL_LENGTH, BASELINE, PRINCIPAL_POINT)
        sys.exit()  # Use sys.exit() instead of exit()

    except ModuleNotFoundError:
        print("Spinnaker SDK not installed. Skipping FLIR cameras.")
    except SystemExit:
        pass
    except ValueError as e:
        pass

    # If no cameras are detected, run the loading functionality
    ui_controller.start()
