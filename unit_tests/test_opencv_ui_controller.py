import unittest
import os
from unittest.mock import patch
from ui_objects.opencv_ui_controller import OpencvUIController

class TestOpencvUIController(unittest.TestCase):
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_get_starting_index(self, mock_listdir, mock_exists):
        # Setup
        mock_exists.return_value = True
        mock_listdir.return_value = ['image1.png', 'image2.png', 'image10.png']
        controller = OpencvUIController('test_prefix', 1.0, 1.0)

        # Test
        starting_index = controller._get_starting_index('some_directory')

        # Assert
        self.assertEqual(starting_index, 11)

    @patch('os.path.exists')
    def test_get_starting_index_no_files(self, mock_exists):
        # Setup
        mock_exists.return_value = False
        controller = OpencvUIController('test_prefix', 1.0, 1.0)

        # Test
        starting_index = controller._get_starting_index('some_directory')

        # Assert
        self.assertEqual(starting_index, 1)

if __name__ == '__main__':
    unittest.main()
