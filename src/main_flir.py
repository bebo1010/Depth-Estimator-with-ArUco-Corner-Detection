"""
Main function to start the application with FLIR camera system.
"""

if __name__ == "__main__":
    from .ui_objects import OpencvUIController
    from .camera_objects import FlirCameraSystem, DualFlirSystem

    FOCAL_LENGTH = 1060  # in pixels
    BASELINE = 80  # in mm
    PRINCIPAL_POINT = (640, 360)  # unchecked in pixels

    UI = OpencvUIController(system_prefix="GH3",
                             focal_length=FOCAL_LENGTH,
                             baseline=BASELINE,
                             principal_point=PRINCIPAL_POINT)

    CONFIG = "./src/camera_config/GH3_camera_config.yaml"
    SN1 = "21091478"
    SN2 = "21091470"

    camera1 = FlirCameraSystem(CONFIG, SN1)
    camera2 = FlirCameraSystem(CONFIG, SN2)
    cameras = DualFlirSystem(camera1, camera2)
    UI.set_camera_system(cameras)

    UI.start()
