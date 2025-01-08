"""
Main function to start the application with Realsense camera system.
"""

if __name__ == "__main__":
    from .ui_objects import OpencvUIController
    from .camera_objects import RealsenseCameraSystem

    # D415
    FOCAL_LENGTH = 908.36  # in pixels
    BASELINE = 55  # in mm
    WIDTH = 1280
    HEIGHT = 720
    PRINCIPAL_POINT = (614.695, 354.577)  # in pixels

    # D435
    # FOCAL_LENGTH = 425.203  # in pixels
    # BASELINE = 50  # in mm
    # WIDTH = 848
    # HEIGHT = 480
    # PRINCIPAL_POINT = (424, 240)  # unchecked in pixels

    UI = OpencvUIController(system_prefix="Realsense",
                            focal_length=FOCAL_LENGTH,
                            baseline=BASELINE,
                            principal_point=PRINCIPAL_POINT)

    cameras = RealsenseCameraSystem(width=WIDTH, height=HEIGHT)
    UI.set_camera_system(cameras)

    UI.start()
