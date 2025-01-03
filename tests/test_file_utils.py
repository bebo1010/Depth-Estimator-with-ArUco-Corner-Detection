"""
Unit tests for the file_utils module.
"""

import os
import logging
import unittest
import coverage
from unittest.mock import patch, mock_open, call

import yaml
import numpy as np

from src.utils import get_starting_index, parse_yaml_config, setup_directories, setup_logging
from src.utils.file_utils import save_images

class TestFileUtils(unittest.TestCase):
    """
    Test suite for the file_utils module.
    """
    def setUp(self):
        """
        Set up the test environment before each test.
        """
        logging.disable(logging.CRITICAL)  # Suppress log messages below CRITICAL level

    def tearDown(self):
        """
        Clean up the test environment after each test.
        """
        logging.disable(logging.NOTSET)  # Re-enable logging after tests

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_get_starting_index(self, mock_listdir, mock_exists):
        """
        Test retrieval of the starting index when files exist.
        """
        mock_exists.return_value = True
        mock_listdir.return_value = ['image1.png', 'image2.png', 'image10.png']
        self.assertEqual(get_starting_index('some_directory'), 11)

    @patch('os.path.exists')
    def test_get_starting_index_no_files(self, mock_exists):
        """
        Test retrieval of the starting index when no files exist.
        """
        mock_exists.return_value = False
        self.assertEqual(get_starting_index('some_directory'), 1)

    def test_parse_yaml_config(self):
        """
        Test parsing of YAML configuration file.
        """
        yaml_content = """
        camera_settings:
            width: 1920
            height: 1084
        """
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            config = parse_yaml_config("dummy_path")
            self.assertEqual(config['camera_settings']['width'], 1920)
            self.assertEqual(config['camera_settings']['height'], 1084)

    def test_parse_yaml_config_oserror(self):
        """
        Test parsing of YAML configuration file when OSError is raised.
        """
        with patch('builtins.open', side_effect=OSError("File not found")):
            config = parse_yaml_config("dummy_path")
            self.assertIsNone(config)

    def test_parse_yaml_config_yamlerror(self):
        """
        Test parsing of YAML configuration file when YAMLError is raised.
        """
        with patch('builtins.open', mock_open(read_data="invalid_yaml: [")):
            with patch('yaml.safe_load', side_effect=yaml.YAMLError("YAML error")):
                config = parse_yaml_config("dummy_path")
                self.assertIsNone(config)

    @patch('os.makedirs')
    def test_setup_directories(self, mock_makedirs):
        """
        Test the creation of directories.
        """
        base_dir = "test_base_dir"
        setup_directories(base_dir)
        expected_calls = [
            call(os.path.join(base_dir, "left_images"), exist_ok=True),
            call(os.path.join(base_dir, "right_images"), exist_ok=True),
            call(os.path.join(base_dir, "depth_images"), exist_ok=True),
            call(os.path.join(base_dir, "left_chessboard_images"), exist_ok=True),
            call(os.path.join(base_dir, "right_chessboard_images"), exist_ok=True)
        ]
        mock_makedirs.assert_has_calls(expected_calls, any_order=True)

    @patch('logging.basicConfig')
    def test_setup_logging(self, mock_basic_config):
        """
        Test the setup of logging.
        """
        base_dir = "test_base_dir"
        setup_logging(base_dir)
        log_path = os.path.join(base_dir, "aruco_depth_log.txt")
        mock_basic_config.assert_called_once_with(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    @patch('cv2.imwrite')
    @patch('os.path.join', side_effect=lambda *args: "/".join(args))
    def test_save_images_left_right_only(self, _mock_path_join, mock_cv2_imwrite):
        """
        Test saving left and right images only.
        """
        base_dir = "./test_base_dir"
        left_image = np.zeros((10, 10), dtype=np.uint8)
        right_image = np.zeros((10, 10), dtype=np.uint8)
        image_index = 1
        prefix = "test"

        save_images(base_dir, left_image, right_image, image_index,
                    prefix=prefix)
        mock_cv2_imwrite.assert_any_call(f"./test_base_dir/left_{prefix}_images/left_image1.png", left_image)
        mock_cv2_imwrite.assert_any_call(f"./test_base_dir/right_{prefix}_images/right_image1.png", right_image)

    @patch('cv2.imwrite')
    @patch('numpy.save')
    @patch('os.path.join', side_effect=lambda *args: "/".join(args))
    def test_save_images_with_first_depth(self, _mock_path_join, mock_npy_save, mock_cv2_imwrite):
        """
        Test saving left, right, and first depth image.
        """
        base_dir = "./test_base_dir"
        left_image = np.zeros((10, 10), dtype=np.uint8)
        right_image = np.zeros((10, 10), dtype=np.uint8)
        depth_image1 = np.zeros((10, 10), dtype=np.uint16)
        image_index = 1
        prefix = "test"

        save_images(base_dir, left_image, right_image, image_index,
                    first_depth_image=depth_image1, prefix=prefix)
        mock_cv2_imwrite.assert_any_call(f"./test_base_dir/left_{prefix}_images/left_image1.png", left_image)
        mock_cv2_imwrite.assert_any_call(f"./test_base_dir/right_{prefix}_images/right_image1.png", right_image)

        mock_cv2_imwrite.assert_any_call("./test_base_dir/depth_images/depth_image1_1.png", depth_image1)
        mock_npy_save.assert_any_call("./test_base_dir/depth_images/depth_image1_1.npy", depth_image1)

    @patch('cv2.imwrite')
    @patch('numpy.save')
    @patch('os.path.join', side_effect=lambda *args: "/".join(args))
    def test_save_images_with_first_and_second_depth(self, _mock_path_join, mock_npy_save, mock_cv2_imwrite):
        """
        Test saving left, right, first, and second depth images.
        """
        base_dir = "./test_base_dir"
        left_image = np.zeros((10, 10), dtype=np.uint8)
        right_image = np.zeros((10, 10), dtype=np.uint8)
        depth_image1 = np.zeros((10, 10), dtype=np.uint16)
        depth_image2 = np.zeros((10, 10), dtype=np.uint16)
        image_index = 1
        prefix = "test"

        save_images(base_dir, left_image, right_image, image_index,
                    first_depth_image=depth_image1, second_depth_image=depth_image2,
                    prefix=prefix)
        mock_cv2_imwrite.assert_any_call(f"./test_base_dir/left_{prefix}_images/left_image1.png", left_image)
        mock_cv2_imwrite.assert_any_call(f"./test_base_dir/right_{prefix}_images/right_image1.png", right_image)

        mock_cv2_imwrite.assert_any_call("./test_base_dir/depth_images/depth_image1_1.png", depth_image1)
        mock_npy_save.assert_any_call("./test_base_dir/depth_images/depth_image1_1.npy", depth_image1)

        mock_cv2_imwrite.assert_any_call("./test_base_dir/depth_images/depth_image2_1.png", depth_image2)
        mock_npy_save.assert_any_call("./test_base_dir/depth_images/depth_image2_1.npy", depth_image2)

if __name__ == '__main__':
    cov = coverage.Coverage()
    cov.start()

    unittest.main()

    cov.stop()
    cov.save()

    cov.html_report()
    print("Done.")