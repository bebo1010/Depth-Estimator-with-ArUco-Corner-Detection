"""
Main function to start the application with FLIR camera system.
"""

if __name__ == "__main__":
    from ui_objects.opencv_ui_controller import OpencvUIController
    from camera_objects.two_cameras.dual_flir_system import DualFlirSystem
    from camera_objects.single_camera.flir_camera_system import FlirCameraSystem

    FOCAL_LENGTH = 1060  # in pixels
    BASELINE = 80  # in mm

    UI = OpencvUIController(system_prefix="GH3", focal_length=FOCAL_LENGTH, baseline=BASELINE)

    CONFIG = "./camera_config/GH3_camera_config.yaml"
    SN1 = "21091478"
    SN2 = "21091470"

    camera1 = FlirCameraSystem(CONFIG, SN1)
    camera2 = FlirCameraSystem(CONFIG, SN2)
    cameras = DualFlirSystem(camera1, camera2)
    UI.set_camera_system(cameras)

    UI.start()
