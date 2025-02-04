if __name__ == "__main__":
    from .ui_objects import OpencvUIController
    
    # Initialize the UI controller with dummy parameters
    ui_controller = OpencvUIController(system_prefix="dummy", focal_length=1.0, baseline=1.0, principal_point=(0, 0))
    
    # Start the UI controller
    ui_controller.start()
    