"""
Unit tests for the file_utils module.
"""

import logging
import unittest
from unittest.mock import patch, mock_open

import yaml

from src.utils.file_utils import get_starting_index, parse_yaml_config

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

if __name__ == '__main__':
    unittest.main()
