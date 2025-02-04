"""
Unit tests for the file_utils module.
"""

import os
import logging

from unittest.mock import patch, mock_open, call
import unittest
import coverage

import yaml
import numpy as np

from src.utils import get_starting_index, parse_yaml_config, setup_directories, setup_logging
from src.utils.file_utils import save_images, load_images_from_directory, save_setup_info, load_setup_info

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
            call(os.path.join(base_dir, "left_ArUco_images"), exist_ok=True),
            call(os.path.join(base_dir, "right_ArUco_images"), exist_ok=True),
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

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_load_images_from_directory_valid(self, mock_listdir, mock_exists):
        """
        Test loading images from a valid directory structure.
        """
        mock_exists.side_effect = lambda path: True
        mock_listdir.side_effect = lambda path: {
            "left_ArUco_images": ["left_image1.png", "left_image2.png"],
            "right_ArUco_images": ["right_image1.png", "right_image2.png"],
            "depth_images": ["depth_image1_1.npy", "depth_image2_1.npy",
                             "depth_image1_2.npy", "depth_image2_2.npy"]
        }[os.path.basename(path)]

        loaded_images, error = load_images_from_directory("test_directory")
        self.assertIsNone(error)
        self.assertEqual(len(loaded_images), 2)
        self.assertEqual(loaded_images[0], (
            os.path.join("test_directory", "left_ArUco_images", "left_image1.png"),
            os.path.join("test_directory", "right_ArUco_images", "right_image1.png"),
            os.path.join("test_directory", "depth_images", "depth_image1_1.npy"),
            os.path.join("test_directory", "depth_images", "depth_image2_1.npy")
        ))
        self.assertEqual(loaded_images[1], (
            os.path.join("test_directory", "left_ArUco_images", "left_image2.png"),
            os.path.join("test_directory", "right_ArUco_images", "right_image2.png"),
            os.path.join("test_directory", "depth_images", "depth_image1_2.npy"),
            os.path.join("test_directory", "depth_images", "depth_image2_2.npy")
        ))

    @patch('os.path.exists')
    def test_load_images_from_directory_invalid_structure(self, mock_exists):
        """
        Test loading images from an invalid directory structure.
        """
        mock_exists.side_effect = lambda path: "left_ArUco_images" in path

        loaded_images, error = load_images_from_directory("test_directory")
        self.assertIsNotNone(error)
        self.assertIsNone(loaded_images)
        self.assertEqual(error, "Invalid directory structure.")

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_load_images_from_directory_mismatched_counts(self, mock_listdir, mock_exists):
        """
        Test loading images when left and right image counts are mismatched.
        """
        mock_exists.side_effect = lambda path: True
        mock_listdir.side_effect = lambda path: {
            "left_ArUco_images": ["left_image1.png"],
            "right_ArUco_images": ["right_image1.png", "right_image2.png"],
            "depth_images": []
        }[os.path.basename(path)]

        loaded_images, error = load_images_from_directory("test_directory")
        self.assertIsNotNone(error)
        self.assertIsNone(loaded_images)
        self.assertEqual(error, "No images found or mismatched image counts.")

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_load_images_from_directory_no_depth_images(self, mock_listdir, mock_exists):
        """
        Test loading images when no depth images are present.
        """
        mock_exists.side_effect = lambda path: True
        mock_listdir.side_effect = lambda path: {
            "left_ArUco_images": ["left_image1.png", "left_image2.png"],
            "right_ArUco_images": ["right_image1.png", "right_image2.png"],
            "depth_images": []
        }[os.path.basename(path)]

        loaded_images, error = load_images_from_directory("test_directory")
        self.assertIsNone(error)
        self.assertEqual(len(loaded_images), 2)
        self.assertEqual(loaded_images[0], (
            os.path.join("test_directory", "left_ArUco_images", "left_image1.png"),
            os.path.join("test_directory", "right_ArUco_images", "right_image1.png"),
            None, None
        ))

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_setup_info(self, mock_json_dump, mock_open_file):
        """
        Test saving setup information to a JSON file.
        """
        base_dir = "test_base_dir"
        camera_params = {
            'system_prefix': 'test_base_dir',
            'focal_length': 1.0,
            'baseline': 1.0,
            'principal_point': (0, 0),
            'width': 1920,
            'height': 1080
        }

        save_setup_info(base_dir, camera_params)

        setup_info = {
            "system_prefix": "test_base_dir",
            "focal_length": 1.0,
            "baseline": 1.0,
            "width": 1920,
            "height": 1080,
            "principal_point": (0, 0)
        }
        mock_open_file.assert_called_once_with(os.path.join(base_dir, "setup.json"), 'w', encoding="utf-8")
        mock_json_dump.assert_called_once_with(setup_info, mock_open_file(), indent=4)

    @patch('os.path.exists')
    def test_load_setup_info(self, mock_exists):
        """
        Test loading setup information from a JSON file.
        """
        mock_exists.return_value = True

        read_data = (
            '{"system_prefix": "test_base_dir", "focal_length": 1.0, "baseline": 1.0, '
            '"width": 1920, "height": 1080, "principal_point": [0, 0]}'
        )


        directory = "test_directory"
        with patch('builtins.open', mock_open(read_data=read_data)):
            setup_info = load_setup_info(directory)


        expected_setup_info = {
            "system_prefix": "test_base_dir",
            "focal_length": 1.0,
            "baseline": 1.0,
            "width": 1920,
            "height": 1080,
            "principal_point": [0, 0]
        }
        self.assertEqual(setup_info, expected_setup_info)

    @patch('os.path.exists')
    def test_load_setup_info_file_not_found(self, mock_exists):
        """
        Test loading setup information when the file is not found.
        """
        mock_exists.return_value = False

        directory = "test_directory"

        setup_info = load_setup_info(directory)
        self.assertIsNone(setup_info)

if __name__ == '__main__':
    cov = coverage.Coverage()
    cov.start()

    unittest.main()

    cov.stop()
    cov.save()

    cov.html_report()
    print("Done.")
