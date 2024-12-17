"""
Unit tests for the file_utils module.
"""

import unittest
from unittest.mock import patch
from utils.file_utils import get_starting_index

class TestFileUtils(unittest.TestCase):
    """
    Test suite for the file_utils module.
    """

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_get_starting_index(self, mock_listdir, mock_exists):
        """
        Test retrieval of the starting index when files exist.
        """
        # Setup
        mock_exists.return_value = True
        mock_listdir.return_value = ['image1.png', 'image2.png', 'image10.png']

        # Test
        starting_index = get_starting_index('some_directory')

        # Assert
        self.assertEqual(starting_index, 11)

    @patch('os.path.exists')
    def test_get_starting_index_no_files(self, mock_exists):
        """
        Test retrieval of the starting index when no files exist.
        """
        # Setup
        mock_exists.return_value = False

        # Test
        starting_index = get_starting_index('some_directory')

        # Assert
        self.assertEqual(starting_index, 1)

if __name__ == '__main__':
    unittest.main()
