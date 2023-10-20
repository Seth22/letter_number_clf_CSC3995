import unittest
import os
import logging
import sys
from preprocess import validation
logger = logging.getLogger()
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

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

    def test_filetypes_clean_files(self):

        if os.path.isfile(f"{pardir}/test_files/val_test_false/p/1885.jpeg"):
            os.remove(f"{pardir}/test_files/val_test_false/p/1885.jpeg")
        self.assertEqual(validation.val_filetypes(f"{pardir}/test_files/val_test_true",filetype=".png"), False)
        self.assertEqual(validation.val_filetypes(f"{pardir}/test_files/val_test_true",filetype="png"),False)
        self.assertEqual(validation.val_filetypes(f"{pardir}/test_files/val_test_false",filetype="png"),False)

        #create a bad file
        os.mknod(f"{pardir}/test_files/val_test_false/p/1885.jpeg")

        self.assertEqual(validation.val_filetypes(f"{pardir}/test_files/val_test_false",filetype="png"),True)

        #remove bad file
        validation.val_remove_filetypes(f"{pardir}/test_files/val_test_false",filetype="png")

        self.assertEqual(validation.val_filetypes(f"{pardir}/test_files/val_test_false",filetype="png"),False)


if __name__ == '__main__':
    unittest.main()
