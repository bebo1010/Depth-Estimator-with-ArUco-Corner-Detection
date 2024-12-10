
from ui_objects.opencv_ui_controller import OpencvUIController
from camera_objects.flir_camera_system import FlirCameraSystem

if __name__ == "__main__":
    focal_length = 1060  # in pixels
    baseline = 80  # in mm

    UI = OpencvUIController(system_prefix="GH3", focal_length=focal_length, baseline=baseline)

    cameras = FlirCameraSystem("./camera_config/GH3_camera_config.yaml")
    UI.set_camera_system(cameras)

    UI.start()
    