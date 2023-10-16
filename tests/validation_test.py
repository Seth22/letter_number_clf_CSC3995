import unittest
import os
import logging
from preprocess import validation

logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    format='%(levelname)s - %(message)s'  # Customize the log message format
)

pardir = os.getcwd()

#Validation functions return an error variable true when there is an error false when there is no error

class MyTestCase(unittest.TestCase):
    def test_val_file_number(self):

        self.assertEqual(validation.val_file_number(f"{pardir}/test_files/val_test_true",20),False)
        self.assertEqual(validation.val_file_number(f"{pardir}/test_files/val_test_false",20),True)

    def test_val_file_location(self):
        self.assertEqual(validation.val_file_location(f"{pardir}/test_files/val_test_true"),False)
        self.assertEqual(validation.val_file_location(f"{pardir}/test_files/val_test_false"), True)


if __name__ == '__main__':
    unittest.main()
