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
        configure_gpio_primary() -> None
        configure_gpio_secondary() -> None
        enable_trigger_mode() -> None
        disable_trigger_mode() -> None
        _get_default_config() -> dict
        _get_serial_number() -> int
        _configure_camera() -> None
        _load_user_set(str) -> None
        _configure_camera_general() -> None
        _configure_acquisition() -> None
        _configure_exposure() -> None
        _configure_gain() -> None
        _configure_white_balance() -> None
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

        self._configure_camera()

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

    def configure_gpio_primary(self) -> None:
        """
        Configure the GPIO settings to primary for single camera.
        Settings: trigger mode, line selector, line mode, line source

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = self.cam.GetNodeMap()
        gpio_primary_config = self.full_config['gpio_primary']
        trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if PySpin.IsWritable(trigger_mode):
            trigger_mode_on = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName(gpio_primary_config['trigger_mode']))
            trigger_mode.SetIntValue(trigger_mode_on.GetValue())
            logging.info('Trigger mode of primary camera %s is set to %s',
                         serial_number, gpio_primary_config['trigger_mode'])

        line_selector = PySpin.CEnumerationPtr(nodemap.GetNode('LineSelector'))
        if PySpin.IsWritable(line_selector):
            line_selector_entry = \
                PySpin.CEnumEntryPtr(line_selector.GetEntryByName(gpio_primary_config['line_selector']))
            line_selector.SetIntValue(line_selector_entry.GetValue())
            logging.info('Line selector of primary camera %s is set to %s',
                         serial_number, gpio_primary_config['line_selector'])

        line_mode = PySpin.CEnumerationPtr(nodemap.GetNode('LineMode'))
        if PySpin.IsWritable(line_mode):
            line_mode_entry = PySpin.CEnumEntryPtr(line_mode.GetEntryByName(gpio_primary_config['line_mode']))
            line_mode.SetIntValue(line_mode_entry.GetValue())
            logging.info('Line mode of primary camera %s is set to %s',
                         serial_number, gpio_primary_config['line_mode'])

        line_source = PySpin.CEnumerationPtr(nodemap.GetNode('LineSource'))
        if PySpin.IsWritable(line_source):
            line_source_entry = PySpin.CEnumEntryPtr(line_source.GetEntryByName(gpio_primary_config['line_source']))
            line_source.SetIntValue(line_source_entry.GetValue())
            logging.info('Line source of primary camera %s is set to %s',
                         serial_number, gpio_primary_config['line_source'])

    def configure_gpio_secondary(self) -> None:
        """
        Configure the GPIO settings to secondary for single camera.
        Settings: trigger selector, trigger mode, trigger source, trigger overlap, trigger source

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = self.cam.GetNodeMap()
        gpio_secondary_config = self.full_config['gpio_secondary']

        trigger_selector = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSelector'))
        if PySpin.IsWritable(trigger_selector):
            trigger_selector_entry = \
                PySpin.CEnumEntryPtr(trigger_selector.GetEntryByName(gpio_secondary_config['trigger_selector']))
            trigger_selector.SetIntValue(trigger_selector_entry.GetValue())
            logging.info('Trigger selector of secondary camera %s is set to %s',
                         serial_number, gpio_secondary_config['trigger_selector'])

        trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if PySpin.IsWritable(trigger_mode):
            trigger_mode_on = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName(gpio_secondary_config['trigger_mode']))
            trigger_mode.SetIntValue(trigger_mode_on.GetValue())
            logging.info('Trigger mode of secondary camera %s is set to %s',
                         serial_number, gpio_secondary_config['trigger_mode'])

        trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
        if PySpin.IsWritable(trigger_source):
            trigger_source_entry = \
                PySpin.CEnumEntryPtr(trigger_source.GetEntryByName(gpio_secondary_config['trigger_source']))
            trigger_source.SetIntValue(trigger_source_entry.GetValue())
            logging.info('Trigger source of secondary camera %s is set to %s',
                         serial_number, gpio_secondary_config['trigger_source'])

        trigger_activation = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerActivation'))
        if PySpin.IsWritable(trigger_activation):
            trigger_activation_entry = \
                PySpin.CEnumEntryPtr(trigger_activation.GetEntryByName(gpio_secondary_config['trigger_activation']))
            trigger_activation.SetIntValue(trigger_activation_entry.GetValue())
            logging.info('Trigger activation of secondary camera %s is set to %s',
                         serial_number, gpio_secondary_config['trigger_activation'])

        trigger_overlap = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerOverlap'))
        if PySpin.IsWritable(trigger_overlap):
            trigger_overlap_entry = \
                PySpin.CEnumEntryPtr(trigger_overlap.GetEntryByName(gpio_secondary_config['trigger_overlap']))
            trigger_overlap.SetIntValue(trigger_overlap_entry.GetValue())
            logging.info('Trigger overlap of secondary camera %s is set to %s',
                         serial_number, gpio_secondary_config['trigger_overlap'])

        line_selector = PySpin.CEnumerationPtr(nodemap.GetNode('LineSelector'))
        if PySpin.IsWritable(line_selector):
            line_selector_entry = \
                PySpin.CEnumEntryPtr(line_selector.GetEntryByName(gpio_secondary_config['trigger_source']))
            line_selector.SetIntValue(line_selector_entry.GetValue())
            logging.info('Line selector of secondary camera %s is set to %s',
                         serial_number, gpio_secondary_config['trigger_source'])

    def enable_trigger_mode(self) -> None:
        """
        Enable trigger mode for single camera.

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = self.cam.GetNodeMap()
        trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        trigger_mode_on = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName('On'))
        trigger_mode.SetIntValue(trigger_mode_on.GetValue())
        logging.info('Trigger mode of camera %s is enabled', serial_number)

    def disable_trigger_mode(self) -> None:
        """
        Disable trigger mode for single camera.

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = self.cam.GetNodeMap()
        trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        trigger_mode_off = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName('Off'))
        trigger_mode.SetIntValue(trigger_mode_off.GetValue())
        logging.info('Trigger mode of camera %s is disabled', serial_number)

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
                'trigger_activation': 'FallingEdge',
                'trigger_overlap': 'ReadOut'
            }
        }
        return config

    def _get_serial_number(self) -> int:
        """
        Get serial number for the camera.

        Returns:
            int:
                - int: Serial number of the camera.
        """
        nodemap: PySpin.NodeMap = self.cam.GetTLDeviceNodeMap()
        serial_number_node = PySpin.CStringPtr(nodemap.GetNode('DeviceSerialNumber'))
        if PySpin.IsReadable(serial_number_node):
            return serial_number_node.GetValue()
        return "Unknown"

    def _configure_camera(self) -> None:
        """
        Configure the basic settings for single camera.

        Returns:
            None
        """
        serial_number = self.serial_number
        logging.info("Configuring camera %s", serial_number)

        self._load_user_set()

        self._configure_camera_general()
        self._configure_acquisition()
        self._configure_exposure()
        self._configure_gain()
        self._configure_white_balance()

    def _load_user_set(self, user_set_name: str = "Default") -> None:
        """
        Load a specified user set from the camera.

        args:
            user_set_name (str): The name of the user set to load (e.g., "Default").
        returns:
        No return.
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = self.cam.GetNodeMap()

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

    def _configure_camera_general(self) -> None:
        """
        Configure the general settings for single camera.
        Settings: width, height, offset, pixel format

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = self.cam.GetNodeMap()
        general_config = self.full_config['camera_settings']
        cam_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
        if PySpin.IsWritable(cam_width):
            cam_width.SetValue(general_config['width'])
            logging.info('Width of camera %s is set to %d', serial_number, general_config['width'])

        cam_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        if PySpin.IsWritable(cam_height):
            cam_height.SetValue(general_config['height'])
            logging.info('Height of camera %s is set to %d', serial_number, general_config['height'])

        cam_offset_x = PySpin.CIntegerPtr(nodemap.GetNode('OffsetX'))
        if PySpin.IsWritable(cam_offset_x):
            cam_offset_x.SetValue(general_config['offset_x'])
            logging.info('OffsetX of camera %s is set to %d', serial_number, general_config['offset_x'])

        cam_offset_y = PySpin.CIntegerPtr(nodemap.GetNode('OffsetY'))
        if PySpin.IsWritable(cam_offset_y):
            cam_offset_y.SetValue(general_config['offset_y'])
            logging.info('OffsetY of camera %s is set to %d', serial_number, general_config['offset_y'])

        pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
        if PySpin.IsWritable(pixel_format):
            pixel_format_entry: PySpin.CEnumEntryPtr =  \
                pixel_format.GetEntryByName(general_config['pixel_format'])
            pixel_format.SetIntValue(pixel_format_entry.GetValue())
            logging.info('Pixel format of camera %s is set to %s',
                         serial_number, general_config['pixel_format'])

    def _configure_acquisition(self) -> None:
        """
        Configure the acquisition settings for single camera.
        Settings: continuous streaming, frame rate control

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = self.cam.GetNodeMap()
        acquisition_config = self.full_config['acquisition_settings']

        acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if PySpin.IsWritable(acquisition_mode):
            continuous_mode = PySpin.CEnumEntryPtr(acquisition_mode.GetEntryByName('Continuous'))
            acquisition_mode.SetIntValue(continuous_mode.GetValue())
            logging.info("Acquisition mode of camera %s is set to Continuous", serial_number)

        frame_rate_auto = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionFrameRateAuto'))
        if PySpin.IsWritable(frame_rate_auto):
            frame_rate_auto_off = PySpin.CEnumEntryPtr(frame_rate_auto.GetEntryByName('Off'))
            frame_rate_auto.SetIntValue(frame_rate_auto_off.GetValue())
            logging.info("Frame rate auto of camera %s is set to Off", serial_number)

        frame_rate_enable = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnabled'))
        if PySpin.IsWritable(frame_rate_enable):
            frame_rate_enable.SetValue(True)
            logging.info("Frame rate control of camera %s is enabled", serial_number)

        frame_rate = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
        if PySpin.IsWritable(frame_rate):
            frame_rate.SetValue(acquisition_config['fps'])
            logging.info('Frame rate of camera %s is set to %s fps',
                         serial_number, acquisition_config['fps'])

    def _configure_exposure(self) -> None:
        """
        Configure the exposure settings for single camera.
        Settings: disable automatic exposure, set exposure time

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = self.cam.GetNodeMap()
        exposure_config = self.full_config['exposure_settings']

        exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        if PySpin.IsWritable(exposure_auto):
            exposure_auto_off = PySpin.CEnumEntryPtr(exposure_auto.GetEntryByName('Off'))
            exposure_auto.SetIntValue(exposure_auto_off.GetValue())
            logging.info("Exposure auto of camera %s is set to Off", serial_number)

        exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        if PySpin.IsWritable(exposure_time):
            exposure_time.SetValue(exposure_config['exposure_time'])
            logging.info('Exposure time of camera %s is set to %s',
                         serial_number, exposure_config['exposure_time'])

    def _configure_gain(self) -> None:
        """
        Configure the gain settings for single camera.
        Settings: disable automatic gain, set gain db

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = self.cam.GetNodeMap()
        gain_config = self.full_config['gain_settings']

        gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        if PySpin.IsWritable(gain_auto):
            gain_auto_once =  \
                PySpin.CEnumEntryPtr(gain_auto.GetEntryByName(gain_config['gain_auto']))
            gain_auto.SetIntValue(gain_auto_once.GetValue())
            logging.info('Gain auto of camera %s is set to %s',
                         serial_number, gain_config['gain_auto'])

        gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        if PySpin.IsWritable(gain):
            gain.SetValue(gain_config['gain_value'])
            logging.info('Gain of camera %s is set to %s',
                         serial_number, gain_config['gain_value'])

    def _configure_white_balance(self) -> None:
        """
        Configure the white balance settings for single camera.
        Can also set white balance to `Once`.
        Settings: disable automatic white balance, set white balance red and blue ratio.

        Returns:
            None
        """
        serial_number = self.serial_number
        nodemap: PySpin.NodeMap = self.cam.GetNodeMap()
        white_balance_config = self.full_config['white_balance_settings']
        node_balance_white_auto = PySpin.CEnumerationPtr(nodemap.GetNode('BalanceWhiteAuto'))
        if PySpin.IsWritable(node_balance_white_auto):
            node_balance_white_auto_value =  \
                PySpin.CEnumEntryPtr(node_balance_white_auto.GetEntryByName(white_balance_config['white_balance_auto']))
            node_balance_white_auto.SetIntValue(node_balance_white_auto_value.GetValue())
            logging.info('White balance of camera %s is set to %s',
                         serial_number, white_balance_config['white_balance_auto'])

        if white_balance_config['white_balance_auto'] == "Off":
            node_balance_ratio_selector = PySpin.CEnumerationPtr(nodemap.GetNode('BalanceRatioSelector'))
            if PySpin.IsWritable(node_balance_ratio_selector):
                node_balance_ratio_selector_blue = \
                    PySpin.CEnumEntryPtr(node_balance_ratio_selector.GetEntryByName('Blue'))
                node_balance_ratio_selector.SetIntValue(node_balance_ratio_selector_blue.GetValue())

                node_balance_ratio = PySpin.CFloatPtr(nodemap.GetNode('BalanceRatio'))
                if PySpin.IsWritable(node_balance_ratio):
                    node_balance_ratio.SetValue(white_balance_config['white_balance_blue_ratio'])
                    logging.info('White balance blue ratio of camera %s is set to %f.',
                                 serial_number, white_balance_config['white_balance_blue_ratio'])

                node_balance_ratio_selector_red = \
                    PySpin.CEnumEntryPtr(node_balance_ratio_selector.GetEntryByName('Red'))
                node_balance_ratio_selector.SetIntValue(node_balance_ratio_selector_red.GetValue())
                node_balance_ratio.SetValue(white_balance_config['white_balance_red_ratio'])
                logging.info('White balance red ratio of camera %s is set to %f.',
                             serial_number, white_balance_config['white_balance_red_ratio'])
