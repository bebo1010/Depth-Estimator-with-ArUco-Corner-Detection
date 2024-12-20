"""
Module for FLIR camera system.
"""
import logging
from typing import Tuple

import cv2
import numpy as np
import PySpin

from camera_objects.single_camera.single_camera_system import SingleCameraSystem
from utils.file_utils import parse_yaml_config

class FlirCameraSystem(SingleCameraSystem):
    """
    FLIR camera system, inherited from SingleCameraSystem.

    Functions:
        __init__() -> None
        get_grayscale_image() -> Tuple[bool, np.ndarray]
        get_depth_image() -> Tuple[bool, np.ndarray]
        get_width() -> int
        get_height() -> int
        release() -> bool
        _get_default_config() -> dict
        _get_serial_number(PySpin.CameraPtr) -> int
        _configure_camera(cam: PySpin.CameraPtr) -> None
        _load_user_set(PySpin.CameraPtr, str) -> None
        _configure_camera_general(PySpin.CameraPtr, dict) -> None
        _configure_acquisition(PySpin.CameraPtr, dict) -> None
        _configure_exposure(PySpin.CameraPtr, dict) -> None
        _configure_gain(PySpin.CameraPtr, dict) -> None
        _configure_white_balance(PySpin.CameraPtr, dict) -> None
        _configure_gpio_primary(PySpin.CameraPtr, dict) -> None
        _configure_gpio_secondary(PySpin.CameraPtr, dict) -> None
        _enable_trigger_mode(PySpin.CameraPtr) -> None
        _disable_trigger_mode(PySpin.CameraPtr) -> None
        _configure_master_camera(PySpin.CameraPtr) -> None
        _configure_slave_camera(PySpin.CameraPtr) -> None
    """
    def __init__(self,
                 config_yaml_path,
                 serial_number = "21091478"):
        """
        Initialize FLIR camera system.

        args:
        config_yaml_path (str): path to config file.
        serial_number (str): serial number of the camera.

        returns:
        No return.
        """
        super().__init__()
        self.full_config = parse_yaml_config(config_yaml_path)
        if self.full_config is None:
            self.full_config = self._get_default_config()

        # Get camera and nodemap
        self.system: PySpin.System = PySpin.System.GetInstance()
        self.cam_list: PySpin.CameraList = self.system.GetCameras()

        camera_count = self.cam_list.GetSize()
        if camera_count < 1:
            logging.error("No cameras detected.")
            self.system.ReleaseInstance()
            raise ValueError("No cameras detected.")

        self.cam: PySpin.CameraPtr = self.cam_list.GetBySerial(serial_number)
        self.serial_number = serial_number
        self.cam.Init()

        self._configure_camera(self.cam)

        self._disable_trigger_mode(self.cam)

        self.image_processor = PySpin.ImageProcessor()
        self.image_processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

    def get_grayscale_image(self) -> Tuple[bool, np.ndarray]:
        """
        Get grayscale image for the camera.

        args:
        No arguments.

        returns:
        Tuple[bool, np.ndarray]:
            - bool: Whether image grabbing is successful or not.
            - np.ndarray: grayscale image.
        """
        if not self.cam.IsStreaming():
            self.cam.BeginAcquisition()

        serial_number = self.serial_number

        logging.info("Reading Frame for %s...", serial_number)
        image_result: PySpin.ImagePtr = self.cam.GetNextImage(1000)
        if image_result.IsIncomplete():
            logging.warning('SN %s: Image incomplete with image status %d',
                            serial_number,
                            image_result.GetImageStatus())
            return False, None

        logging.info("Grabbed Frame for %s", serial_number)

        logging.info("Converting Frame for %s...", serial_number)
        image_converted: PySpin.ImagePtr =  \
            self.image_processor.Convert(image_result, PySpin.PixelFormat_BayerRG8)
        image_data = image_converted.GetNDArray()
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BayerRG2GRAY)
        logging.info("Convertion Frame for %s done", serial_number)

        return True, image_data
    def get_depth_image(self) -> Tuple[bool, np.ndarray]:
        """
        Get depth images for the camera system.

        args:
        No arguments.

        returns:
        Tuple[bool, np.ndarray]:
            - bool: Whether depth image grabbing is successful or not.
            - np.ndarray: depth grayscale image.
        """
        # No depth image in flir camera system
        return False, None
    def get_width(self) -> int:
        """
        Get width for the camera system.

        args:
        No arguments.

        returns:
        int:
            - int: Width of the camera system.
        """
        return int(self.full_config['camera_settings']['width'])
    def get_height(self) -> int:
        """
        Get height for the camera system.

        args:
        No arguments.

        returns:
        int:
            - int: Height of the camera system.
        """
        return int(self.full_config['camera_settings']['height'])
    def release(self) -> bool:
        """
        Release the camera system.

        args:
        No arguments.

        returns:
        bool:
            - bool: Whether releasing is successful or not.
        """
        logging.info("Stopping camera acquisition...")
        self.cam.EndAcquisition()

        logging.info("Releasing camera...")
        self.cam.DeInit()
        logging.info("Camera released.")

        logging.info("Clearing camera list...")
        self.cam_list.Clear()
        logging.info("Camera list cleared.")

        logging.info("Releasing system...")
        self.system.ReleaseInstance()
        logging.info("System released.")

        return True
    def _get_default_config(self) -> dict:
        """
        Get default configuration file for flir camera system.
        Default config is for grasshopper3 cameras

        args:
        No arguments.

        returns:
        dict:
            - dict: dictionary of full configs.
        """
        config = {
            'camera_settings': {
                'width': 1920,
                'height': 1084,
                'offset_x': 0,
                'offset_y': 58,
                'pixel_format': 'BayerRG8'
            },
            'acquisition_settings': {
                'fps': 179
            },
            'device_settings': {
                'device_link_throughput_limit': 380160000
            },
            'exposure_settings': {
                'exposure_auto': False,
                'exposure_mode': 'Timed',
                'exposure_time': 2000
            },
            'gain_settings': {
                'gain_auto': 'Off',
                'gain_value': 15.0
            },
            'white_balance_settings': {
                'white_balance_auto': 'Off',
                'white_balance_red_ratio': 2.0,
                'white_balance_blue_ratio': 3.0
            },
            'gpio_primary': {
                'trigger_mode': 'On',
                'line_selector': 'Line2',
                'line_mode': 'Output',
                'line_source': 'ExposureActive'
            },
            'gpio_secondary': {
                'trigger_selector': 'FrameStart',
                'trigger_mode': 'On',
                'trigger_source': 'Line3',
                'trigger_overlap': 'ReadOut'
            }
        }
        return config

    def _get_serial_number(self, cam: PySpin.CameraPtr) -> int:
        """
        Get serial number for the camera.
        Currently unused, Leave this for future use.

        Args:
            cam (PySpin.CameraPtr): The camera object to configure.

        Returns:
            int:
                - int: Serial number of the camera.
        """
        nodemap: PySpin.NodeMap = cam.GetTLDeviceNodeMap()
        serial_number_node = PySpin.CStringPtr(nodemap.GetNode('DeviceSerialNumber'))
        if PySpin.IsReadable(serial_number_node):
            return serial_number_node.GetValue()
        return "Unknown"

    def _configure_camera(self, cam: PySpin.CameraPtr) -> None:
        """
        Configure the basic settings for single camera.

        Args:
            cam (PySpin.CameraPtr): The camera object to configure.

        Returns:
            None
        """
        serial_number = self.serial_number
        logging.info("Configuring camera %s", serial_number)

        self._load_user_set(cam)

        self._configure_camera_general(cam, self.full_config['camera_settings'])
        self._configure_acquisition(cam, self.full_config['acquisition_settings'])
        self._configure_exposure(cam, self.full_config['exposure_settings'])
        self._configure_gain(cam, self.full_config['gain_settings'])
        self._configure_white_balance(cam, self.full_config['white_balance_settings'])

    def _load_user_set(self, cam: PySpin.CameraPtr, user_set_name: str = "Default") -> None:
        """
        Load a specified user set from the camera.

        args:
            cam (PySpin.CameraPtr): The camera object.
            user_set_name (str): The name of the user set to load (e.g., "Default").
        returns:
        No return.
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = cam.GetNodeMap()

        # Select the User Set
        user_set_selector = PySpin.CEnumerationPtr(nodemap.GetNode('UserSetSelector'))
        if not PySpin.IsReadable(user_set_selector) or not PySpin.IsWritable(user_set_selector):
            logging.warning("User Set Selector of camera %s is not accessible", serial_number)
            return

        user_set_entry = user_set_selector.GetEntryByName(user_set_name)
        if not PySpin.IsReadable(user_set_entry):
            logging.warning('User Set %s of camera %s is not available',
                            user_set_name, serial_number)
            return

        user_set_selector.SetIntValue(user_set_entry.GetValue())
        logging.info("User Set %s of camera %s selected", user_set_name, serial_number)

        # Load the User Set
        user_set_load = PySpin.CCommandPtr(nodemap.GetNode('UserSetLoad'))
        if not PySpin.IsWritable(user_set_load):
            logging.warning("User Set Load of camera %s is not executable", serial_number)
            return

        user_set_load.Execute()
        logging.info("User Set %s of camera %s loaded", user_set_name, serial_number)

    def _configure_camera_general(self, cam: PySpin.CameraPtr, general_config: dict) -> None:
        """
        Configure the general settings for single camera.
        Settings: width, height, offset, pixel format

        Args:
            cam (PySpin.CameraPtr): The camera object to configure.
            general_config (dict): The dictionary of general camera settings.

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = cam.GetNodeMap()
        cam_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
        cam_width.SetValue(general_config['width'])
        logging.info('Width of camera %s is set to %d',
                     serial_number, general_config['width'])

        cam_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        cam_height.SetValue(general_config['height'])
        logging.info('Height of camera %s is set to %d',
                     serial_number, general_config['height'])

        cam_offset_x = PySpin.CIntegerPtr(nodemap.GetNode('OffsetX'))
        cam_offset_x.SetValue(general_config['offset_x'])
        logging.info('OffsetX of camera %s is set to %d',
                     serial_number, general_config['offset_x'])

        cam_offset_y = PySpin.CIntegerPtr(nodemap.GetNode('OffsetY'))
        cam_offset_y.SetValue(general_config['offset_y'])
        logging.info('OffsetY of camera %s is set to %d',
                     serial_number, general_config['offset_y'])

        pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
        pixel_format_entry: PySpin.CEnumEntryPtr =  \
            pixel_format.GetEntryByName(general_config['pixel_format'])
        pixel_format.SetIntValue(pixel_format_entry.GetValue())
        logging.info('Pixel format of camera %s is set to %s',
                     serial_number, general_config['pixel_format'])

    def _configure_acquisition(self, cam: PySpin.CameraPtr, acquisition_config: dict) -> None:
        """
        Configure the acquisition settings for single camera.
        Settings: continuous streaming, frame rate control

        Args:
            cam (PySpin.CameraPtr): The camera object to configure.
            acquisition_config (dict): The dictionary of acquisition settings.

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = cam.GetNodeMap()

        acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        continuous_mode = PySpin.CEnumEntryPtr(acquisition_mode.GetEntryByName('Continuous'))
        acquisition_mode.SetIntValue(continuous_mode.GetValue())
        logging.info("Acquisition mode of camera %s is set to Continuous", serial_number)

        frame_rate_auto = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionFrameRateAuto'))
        frame_rate_auto_off = PySpin.CEnumEntryPtr(frame_rate_auto.GetEntryByName('Off'))
        frame_rate_auto.SetIntValue(frame_rate_auto_off.GetValue())
        logging.info("Frame rate auto of camera %s is set to Off", serial_number)

        frame_rate_enable = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnabled'))
        frame_rate_enable.SetValue(True)
        logging.info("Frame rate control of camera %s is enabled", serial_number)

        frame_rate = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
        frame_rate.SetValue(acquisition_config['fps'])
        logging.info('Frame rate of camera %s is set to %s fps',
                     serial_number, acquisition_config['fps'])

    def _configure_exposure(self, cam: PySpin.CameraPtr, exposure_config: dict) -> None:
        """
        Configure the exposure settings for single camera.
        Settings: disable automatic exposure, set exposure time

        Args:
            cam (PySpin.CameraPtr): The camera object to configure.
            exposure_config (dict): The dictionary of exposure settings.

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = cam.GetNodeMap()

        exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        exposure_auto_off = PySpin.CEnumEntryPtr(exposure_auto.GetEntryByName('Off'))
        exposure_auto.SetIntValue(exposure_auto_off.GetValue())
        logging.info("Exposure auto of camera %s is set to Off", serial_number)

        exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        exposure_time.SetValue(exposure_config['exposure_time'])
        logging.info('Exposure time of camera %s is set to %s',
                     serial_number, exposure_config['exposure_time'])

    def _configure_gain(self, cam: PySpin.CameraPtr, gain_config: dict) -> None:
        """
        Configure the gain settings for single camera.
        Settings: disable automatic gain, set gain db

        Args:
            cam (PySpin.CameraPtr): The camera object to configure.
            gain_config (dict): The dictionary of gain settings.

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = cam.GetNodeMap()

        gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        gain_auto_once =  \
            PySpin.CEnumEntryPtr(gain_auto.GetEntryByName(gain_config['gain_auto']))
        gain_auto.SetIntValue(gain_auto_once.GetValue())
        logging.info('Gain auto of camera %s is set to %s',
                     serial_number, gain_config['gain_auto'])

        gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        gain.SetValue(gain_config['gain_value'])
        logging.info('Gain of camera %s is set to %s',
                     serial_number, gain_config['gain_value'])

    def _configure_white_balance(self, cam: PySpin.CameraPtr, white_balance_config: dict) -> None:
        """
        Configure the white balance settings for single camera.
        Can also set white balance to `Once`.
        Settings: disable automatic white balance, set white balance red and blue ratio.

        Args:
            cam (PySpin.CameraPtr): The camera object to configure.
            white_balance_config (dict): The dictionary of white balance settings.

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = cam.GetNodeMap()
        node_balance_white_auto = PySpin.CEnumerationPtr(nodemap.GetNode('BalanceWhiteAuto'))
        node_balance_white_auto_value =  \
            PySpin.CEnumEntryPtr(node_balance_white_auto.GetEntryByName(white_balance_config['white_balance_auto']))
        node_balance_white_auto.SetIntValue(node_balance_white_auto_value.GetValue())
        logging.info('White balance of camera %s is set to %s',
                     serial_number, white_balance_config['white_balance_auto'])

        if white_balance_config['white_balance_auto'] == "Off":
            node_balance_ratio_selector = PySpin.CEnumerationPtr(nodemap.GetNode('BalanceRatioSelector'))
            node_balance_ratio_selector_blue = \
                PySpin.CEnumEntryPtr(node_balance_ratio_selector.GetEntryByName('Blue'))
            node_balance_ratio_selector.SetIntValue(node_balance_ratio_selector_blue.GetValue())

            node_balance_ratio = PySpin.CFloatPtr(nodemap.GetNode('BalanceRatio'))
            node_balance_ratio.SetValue(white_balance_config['white_balance_blue_ratio'])
            logging.info('White balance blue ratio of camera %s is set to %f.',
                         serial_number, white_balance_config['white_balance_blue_ratio'])

            node_balance_ratio_selector_red = PySpin.CEnumEntryPtr(node_balance_ratio_selector.GetEntryByName('Red'))
            node_balance_ratio_selector.SetIntValue(node_balance_ratio_selector_red.GetValue())
            node_balance_ratio.SetValue(white_balance_config['white_balance_red_ratio'])
            logging.info('White balance red ratio of camera %s is set to %f.',
                         serial_number, white_balance_config['white_balance_red_ratio'])

    def _configure_gpio_primary(self, cam: PySpin.CameraPtr, gpio_primary_config: dict) -> None:
        """
        Configure the GPIO settings to primary for single camera.
        Settings: trigger mode, line selector, line mode, line source

        Args:
            cam (PySpin.CameraPtr): The camera object to configure.
            gpio_primary_config (dict): The dictionary of GPIO primary settings.

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = cam.GetNodeMap()
        trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        trigger_mode_on = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName(gpio_primary_config['trigger_mode']))
        trigger_mode.SetIntValue(trigger_mode_on.GetValue())
        logging.info('Trigger mode of primary camera %s is set to %s',
                     serial_number, gpio_primary_config['trigger_mode'])

        line_selector = PySpin.CEnumerationPtr(nodemap.GetNode('LineSelector'))
        line_selector_entry = PySpin.CEnumEntryPtr(line_selector.GetEntryByName(gpio_primary_config['line_selector']))
        line_selector.SetIntValue(line_selector_entry.GetValue())
        logging.info('Line selector of primary camera %s is set to %s',
                     serial_number, gpio_primary_config['line_selector'])

        line_mode = PySpin.CEnumerationPtr(nodemap.GetNode('LineMode'))
        line_mode_entry = PySpin.CEnumEntryPtr(line_mode.GetEntryByName(gpio_primary_config['line_mode']))
        line_mode.SetIntValue(line_mode_entry.GetValue())
        logging.info('Line mode of primary camera %s is set to %s',
                     serial_number, gpio_primary_config['line_mode'])

        line_source = PySpin.CEnumerationPtr(nodemap.GetNode('LineSource'))
        line_source_entry = PySpin.CEnumEntryPtr(line_source.GetEntryByName(gpio_primary_config['line_source']))
        line_source.SetIntValue(line_source_entry.GetValue())
        logging.info('Line source of primary camera %s is set to %s',
                     serial_number, gpio_primary_config['line_source'])

    def _configure_gpio_secondary(self, cam: PySpin.CameraPtr, gpio_secondary_config: dict) -> None:
        """
        Configure the GPIO settings to secondary for single camera.
        Settings: trigger selector, trigger mode, trigger source, trigger overlap, trigger source

        Args:
            cam (PySpin.CameraPtr): The camera object to configure.
            gpio_secondary_config (dict): The dictionary of GPIO secondary settings.

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = cam.GetNodeMap()

        trigger_selector = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSelector'))
        trigger_selector_entry = \
            PySpin.CEnumEntryPtr(trigger_selector.GetEntryByName(gpio_secondary_config['trigger_selector']))
        trigger_selector.SetIntValue(trigger_selector_entry.GetValue())
        logging.info('Trigger selector of secondary camera %s is set to %s',
                     serial_number, gpio_secondary_config['trigger_selector'])

        trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        trigger_mode_on = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName(gpio_secondary_config['trigger_mode']))
        trigger_mode.SetIntValue(trigger_mode_on.GetValue())
        logging.info('Trigger mode of secondary camera %s is set to %s',
                     serial_number, gpio_secondary_config['trigger_mode'])

        trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
        trigger_source_entry = \
            PySpin.CEnumEntryPtr(trigger_source.GetEntryByName(gpio_secondary_config['trigger_source']))
        trigger_source.SetIntValue(trigger_source_entry.GetValue())
        logging.info('Trigger source of secondary camera %s is set to %s',
                     serial_number, gpio_secondary_config['trigger_source'])

        trigger_overlap = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerOverlap'))
        trigger_overlap_entry = \
            PySpin.CEnumEntryPtr(trigger_overlap.GetEntryByName(gpio_secondary_config['trigger_overlap']))
        trigger_overlap.SetIntValue(trigger_overlap_entry.GetValue())
        logging.info('Trigger overlap of secondary camera %s is set to %s',
                     serial_number, gpio_secondary_config['trigger_overlap'])

        line_selector = PySpin.CEnumerationPtr(nodemap.GetNode('LineSelector'))
        line_selector_entry = \
            PySpin.CEnumEntryPtr(line_selector.GetEntryByName(gpio_secondary_config['trigger_source']))
        line_selector.SetIntValue(line_selector_entry.GetValue())
        logging.info('Line selector of secondary camera %s is set to %s',
                     serial_number, gpio_secondary_config['trigger_source'])

    def _enable_trigger_mode(self, cam: PySpin.CameraPtr) -> None:
        """
        Enable trigger mode for single camera.

        Args:
            cam (PySpin.CameraPtr): The camera object to configure.

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = cam.GetNodeMap()
        trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        trigger_mode_on = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName('On'))
        trigger_mode.SetIntValue(trigger_mode_on.GetValue())
        logging.info('Trigger mode of camera %s is enabled', serial_number)

    def _disable_trigger_mode(self, cam: PySpin.CameraPtr) -> None:
        """
        Disable trigger mode for single camera.

        Args:
            cam (PySpin.CameraPtr): The camera object to configure.

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = cam.GetNodeMap()
        trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        trigger_mode_off = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName('Off'))
        trigger_mode.SetIntValue(trigger_mode_off.GetValue())
        logging.info('Trigger mode of camera %s is disabled', serial_number)
