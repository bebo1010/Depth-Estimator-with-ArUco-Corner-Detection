import logging
from typing import Tuple
import yaml

import numpy as np
import PySpin

from .camera_abstract_class import Two_Cameras_System

class Flir_Camera_System(Two_Cameras_System):
    def __init__(self, config_yaml_path, master_serial_number = "21091478", slave_serial_number = "21091470"):
        super().__init__()
        
        self.config = self._parse_yaml_config(config_yaml_path)

        # Get camera and nodemap
        self.system: PySpin.System = PySpin.System.GetInstance()
        self.cam_list: PySpin.CameraList = self.system.GetCameras()

        if self.cam_list.GetSize() == 0:
            print("No cameras detected.")
            self.system.ReleaseInstance()

        self.master_cam: PySpin.CameraPtr = self.cam_list.GetBySerial(master_serial_number)
        self.master_cam.Init()

        self.slave_cam: PySpin.CameraPtr = self.cam_list.GetBySerial(slave_serial_number)
        self.slave_cam.Init()

    def get_grayscale_images(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        # TODO: Finish implementation
        pass
        
    def get_depth_image(self) -> Tuple[bool, np.ndarray]:
        # No depth image in flir camera system
        return False, None

    def get_width(self) -> int:
        return int(self.config['camera_settings']['width'])
    
    def get_height(self) -> int:
        return int(self.config['camera_settings']['height'])
    
    def release(self) -> bool:
        logging.info("Stopping master camera acquisition...")
        self.master_cam.EndAcquisition()

        logging.info("Stopping slave camera acquisition...")
        self.slave_cam.EndAcquisition()

        logging.info("Releasing master camera...")
        self.master_cam.DeInit()
        logging.info("Master camera released.")

        logging.info("Releasing slave camera...")
        self.slave_cam.DeInit()
        logging.info("Slave camera released.")

        logging.info("Clearing camera list...")
        self.cam_list.Clear()
        logging.info("Camera list cleared.")

        logging.info("Releasing system...")
        self.system.ReleaseInstance()
        logging.info("System released.")

        return True

    def _parse_yaml_config(self, config_yaml_path: str) -> dict:
        try:
            with open(config_yaml_path, 'r') as file:
                config = yaml.safe_load(file)
                logging.info(f"Configuration file at {config_yaml_path} successfully loaded")
                return config
        except OSError:
            logging.error(f"Error when loading configuration file at {config_yaml_path}")
            logging.info(f"Fallback to default config")
            config = self._get_default_config()
            return config
        except yaml.YAMLError:
            logging.error(f"Error when parsing yaml in {config_yaml_path}")
            logging.info(f"Fallback to default config")
            config = self._get_default_config()
            return config

    def _get_default_config(self) -> dict:
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

    def _get_serial_number(cam: PySpin.CameraPtr):
        """Retrieve the serial number of the camera."""
        nodemap: PySpin.NodeMap = cam.GetTLDeviceNodeMap()
        serial_number_node = PySpin.CStringPtr(nodemap.GetNode('DeviceSerialNumber'))
        if PySpin.IsReadable(serial_number_node):
            return serial_number_node.GetValue()
        return "Unknown"

    def _configure_camera(self, cam: PySpin.CameraPtr) -> None:
        """Configure the camera settings for both cameras."""
        serial_number = self._get_serial_number(cam)
        logging.info(f"Configuring camera {serial_number}")

        self._configure_camera_general(cam, self.config['camera_settings'])
        self._configure_acquisition(cam, self.config['acquisition_settings'])
        self._configure_exposure(cam, self.config['exposure_settings'])
        self._configure_gain(cam, self.config['gain_settings'])
        self._configure_white_balance(cam, self.config['white_balance_settings'])

    def _configure_camera_general(self, cam: PySpin.CameraPtr, general_config: dict) -> None:
        """Configure general camera settings like width, height, offset, and pixel format."""
        nodemap: PySpin.NodeMap = cam.GetNodeMap()
        cam_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
        cam_width.SetValue(general_config['width'])
        logging.info('Width of camera %s is set to %d', self._get_serial_number(cam), general_config['width'])

        cam_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        cam_height.SetValue(general_config['height'])
        logging.info('Height of camera %s is set to %d', self._get_serial_number(cam), general_config['height'])

        cam_offset_x = PySpin.CIntegerPtr(nodemap.GetNode('OffsetX'))
        cam_offset_x.SetValue(general_config['offset_x'])
        logging.info('OffsetX of camera %s is set to %d', self._get_serial_number(cam), general_config['offset_x'])

        cam_offset_y = PySpin.CIntegerPtr(nodemap.GetNode('OffsetY'))
        cam_offset_y.SetValue(general_config['offset_y'])
        logging.info('OffsetY of camera %s is set to %d', self._get_serial_number(cam), general_config['offset_y'])

        pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
        pixel_format_entry: PySpin.CEnumEntryPtr = pixel_format.GetEntryByName(general_config['pixel_format'])
        pixel_format.SetIntValue(pixel_format_entry.GetValue())
        logging.info('Pixel format of camera %s is set to %s', self._get_serial_number(cam), general_config['pixel_format'])

    def _configure_acquisition(self, cam: PySpin.CameraPtr, acquisition_config: dict) -> None:
        """Configure the camera acquisition settings such as frame rate."""
        serial_number = self._get_serial_number(cam)
        nodemap: PySpin.NodeMap = cam.GetNodeMap()

        acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        continuous_mode = PySpin.CEnumEntryPtr(acquisition_mode.GetEntryByName('Continuous'))
        acquisition_mode.SetIntValue(continuous_mode.GetValue())
        logging.info('Acquisition mode of camera %s is set to Continuous', serial_number)

        frame_rate_auto = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionFrameRateAuto'))
        frame_rate_auto_off = PySpin.CEnumEntryPtr(frame_rate_auto.GetEntryByName('Off'))
        frame_rate_auto.SetIntValue(frame_rate_auto_off.GetValue())
        logging.info('Frame rate auto of camera %s is set to Off', serial_number)

        frame_rate_enable = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnabled'))
        frame_rate_enable.SetValue(True)
        logging.info('Frame rate control of camera %s is enabled', serial_number)

        frame_rate = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
        frame_rate.SetValue(acquisition_config['fps'])
        logging.info('Frame rate of camera %s is set to %s fps', serial_number, acquisition_config['fps'])

    def _configure_exposure(self, cam: PySpin.CameraPtr, exposure_config: dict) -> None:
        """Configure exposure settings like exposure time and auto settings."""
        serial_number = self._get_serial_number(cam)
        nodemap: PySpin.NodeMap = cam.GetNodeMap()

        exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        exposure_auto_off = PySpin.CEnumEntryPtr(exposure_auto.GetEntryByName('Off'))
        exposure_auto.SetIntValue(exposure_auto_off.GetValue())
        logging.info('Exposure auto of camera %s is set to Off', serial_number)

        exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        exposure_time.SetValue(exposure_config['exposure_time'])
        logging.info('Exposure time of camera %s is set to %s', serial_number, exposure_config['exposure_time'])

    def _configure_gain(self, cam: PySpin.CameraPtr, gain_config: dict) -> None:
        """Configure gain settings including auto gain and value."""
        serial_number = self._get_serial_number(cam)
        nodemap: PySpin.NodeMap = cam.GetNodeMap()

        gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        gain_auto_once = PySpin.CEnumEntryPtr(gain_auto.GetEntryByName(gain_config['gain_auto']))
        gain_auto.SetIntValue(gain_auto_once.GetValue())
        logging.info('Gain auto of camera %s is set to %s', serial_number, gain_config['gain_auto'])

        gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        gain.SetValue(gain_config['gain_value'])
        logging.info('Gain of camera %s is set to %s', serial_number, gain_config['gain_value'])

    def _configure_white_balance(self, cam: PySpin.CameraPtr, white_balance_config: dict) -> None:
        """Configure white balance settings including auto and manual adjustments."""
        serial_number = self._get_serial_number(cam)
        nodemap: PySpin.NodeMap = cam.GetNodeMap()
        node_balance_white_auto = PySpin.CEnumerationPtr(nodemap.GetNode('BalanceWhiteAuto'))
        node_balance_white_auto_value = PySpin.CEnumEntryPtr(node_balance_white_auto.GetEntryByName(white_balance_config['white_balance_auto']))
        node_balance_white_auto.SetIntValue(node_balance_white_auto_value.GetValue())
        logging.info('White balance of camera %s is set to %s', serial_number, white_balance_config['white_balance_auto'])

        if white_balance_config['white_balance_auto'] == "Off":
            node_balance_ratio_selector = PySpin.CEnumerationPtr(nodemap.GetNode('BalanceRatioSelector'))
            node_balance_ratio_selector_blue = PySpin.CEnumEntryPtr(node_balance_ratio_selector.GetEntryByName('Blue'))
            node_balance_ratio_selector.SetIntValue(node_balance_ratio_selector_blue.GetValue())

            node_balance_ratio = PySpin.CFloatPtr(nodemap.GetNode('BalanceRatio'))
            node_balance_ratio.SetValue(white_balance_config['white_balance_blue_ratio'])
            logging.info('White balance blue ratio of camera %s is set to %f.', serial_number, white_balance_config['white_balance_blue_ratio'])

            node_balance_ratio_selector_red = PySpin.CEnumEntryPtr(node_balance_ratio_selector.GetEntryByName('Red'))
            node_balance_ratio_selector.SetIntValue(node_balance_ratio_selector_red.GetValue())
            node_balance_ratio.SetValue(white_balance_config['white_balance_red_ratio'])
            logging.info('White balance red ratio of camera %s is set to %f.', serial_number, white_balance_config['white_balance_red_ratio'])

    def _configure_gpio_primary(self, cam: PySpin.CameraPtr, gpio_primary_config: dict) -> None:
        """Configure GPIO settings for the primary camera."""
        serial_number = self._get_serial_number(cam)
        nodemap: PySpin.NodeMap = cam.GetNodeMap()
        trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        trigger_mode_on = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName(gpio_primary_config['trigger_mode']))
        trigger_mode.SetIntValue(trigger_mode_on.GetValue())
        logging.info('Trigger mode of primary camera %s is set to %s', serial_number, gpio_primary_config['trigger_mode'])

        line_selector = PySpin.CEnumerationPtr(nodemap.GetNode('LineSelector'))
        line_selector_entry = PySpin.CEnumEntryPtr(line_selector.GetEntryByName(gpio_primary_config['line_selector']))
        line_selector.SetIntValue(line_selector_entry.GetValue())
        logging.info('Line selector of primary camera %s is set to %s', serial_number, gpio_primary_config['line_selector'])

        line_mode = PySpin.CEnumerationPtr(nodemap.GetNode('LineMode'))
        line_mode_entry = PySpin.CEnumEntryPtr(line_mode.GetEntryByName(gpio_primary_config['line_mode']))
        line_mode.SetIntValue(line_mode_entry.GetValue())
        logging.info('Line mode of primary camera %s is set to %s', serial_number, gpio_primary_config['line_mode'])

        line_source = PySpin.CEnumerationPtr(nodemap.GetNode('LineSource'))
        line_source_entry = PySpin.CEnumEntryPtr(line_source.GetEntryByName(gpio_primary_config['line_source']))
        line_source.SetIntValue(line_source_entry.GetValue())
        logging.info('Line source of primary camera %s is set to %s', serial_number, gpio_primary_config['line_source'])

    def _configure_gpio_secondary(self, cam: PySpin.CameraPtr, gpio_secondary_config: dict) -> None:
        """Configure GPIO settings for the secondary camera."""
        serial_number = self._get_serial_number(cam)
        nodemap: PySpin.NodeMap = cam.GetNodeMap()
        trigger_selector = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSelector'))
        trigger_selector_entry = PySpin.CEnumEntryPtr(trigger_selector.GetEntryByName(gpio_secondary_config['trigger_selector']))
        trigger_selector.SetIntValue(trigger_selector_entry.GetValue())
        logging.info('Trigger selector of secondary camera %s is set to %s', serial_number, gpio_secondary_config['trigger_selector'])

        trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        trigger_mode_on = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName(gpio_secondary_config['trigger_mode']))
        trigger_mode.SetIntValue(trigger_mode_on.GetValue())
        logging.info('Trigger mode of secondary camera %s is set to %s', serial_number, gpio_secondary_config['trigger_mode'])

        trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
        trigger_source_entry = PySpin.CEnumEntryPtr(trigger_source.GetEntryByName(gpio_secondary_config['trigger_source']))
        trigger_source.SetIntValue(trigger_source_entry.GetValue())
        logging.info('Trigger source of secondary camera %s is set to %s', serial_number, gpio_secondary_config['trigger_source'])

        trigger_overlap = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerOverlap'))
        trigger_overlap_entry = PySpin.CEnumEntryPtr(trigger_overlap.GetEntryByName(gpio_secondary_config['trigger_overlap']))
        trigger_overlap.SetIntValue(trigger_overlap_entry.GetValue())
        logging.info('Trigger overlap of secondary camera %s is set to %s', serial_number, gpio_secondary_config['trigger_overlap'])

        line_selector = PySpin.CEnumerationPtr(nodemap.GetNode('LineSelector'))
        line_selector_entry = PySpin.CEnumEntryPtr(line_selector.GetEntryByName(gpio_secondary_config['trigger_source']))
        line_selector.SetIntValue(line_selector_entry.GetValue())
        logging.info('Line selector of secondary camera %s is set to %s', serial_number, gpio_secondary_config['trigger_source'])

    def _enable_trigger_mode(self, cam: PySpin.CameraPtr) -> None:
        """Enable trigger mode for the camera."""
        serial_number = self._get_serial_number(cam)
        nodemap: PySpin.NodeMap = cam.GetNodeMap()
        trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        trigger_mode_on = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName('On'))
        trigger_mode.SetIntValue(trigger_mode_on.GetValue())
        logging.info('Trigger mode of camera %s is enabled', serial_number)

    def _disable_trigger_mode(self, cam: PySpin.CameraPtr) -> None:
        """Disable trigger mode for the camera."""
        serial_number = self._get_serial_number(cam)
        nodemap: PySpin.NodeMap = cam.GetNodeMap()
        trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        trigger_mode_off = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName('Off'))
        trigger_mode.SetIntValue(trigger_mode_off.GetValue())
        logging.info('Trigger mode of camera %s is disabled', serial_number)
