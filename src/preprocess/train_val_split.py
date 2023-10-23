import os, shutil
import random
from . import exception
import logging


def train_val_split(input_dir, output_dir):
    """
    Function:
        Splits **sorted** image files(*with directory name corresponding to label*) into a training and validation set with
        **80%** of images used for training and a random selection(from each class) of **20%** used for validation set
        Outputs training set in *output_dir/ndnmol* and validation set as *output_dir/ndnmol_test_set*
    Important Note:
        Current implementation expects the rows in order to be 0,1,7,8,9,l,m,k,p,n if different labels are needed based
        on row **class_dir** must be changed
    :param input_dir: Expects an input directory with 10 class folders
    :param output_dir: Directory to put training and validation files
    :return: Void
    """
    class_dirs = ['0', '1', '7', '8', '9', 'k', 'l', 'm', 'n', 'p']

    # create target directories ndnmol and ndnmol_test_set and set variables to remember location
    logging.info("Creating train/test directories")
    ndnmol_dir = f"{output_dir}/ndnmol"
    ndnmol_test_dir = f"{output_dir}/ndnmol_test_set"

    os.mkdir(ndnmol_dir)
    os.mkdir(ndnmol_test_dir)

    if input_dir is None:
        raise exception.input_dir_empty("No Input directory is equal to NONE in train_val_split")

    source_dir = input_dir

    # creates class folders in training and test set
    for dirs in class_dirs:
        os.mkdir(f"{ndnmol_dir}/{dirs}")
        os.mkdir(f"{ndnmol_test_dir}/{dirs}")
    logging.info("Adding and splitting files...")
    # moves files from original dataset to our test/train dataset
    for folder in os.listdir(source_dir):
        # randomly selects 20 files to move to test set
        random_files = random.sample(os.listdir(f"{source_dir}/{folder}"), 20)
        for filename in os.listdir(f"{source_dir}/{folder}"):
            # copy all files in class directory from original data to new training set
            shutil.copy(f"{source_dir}/{folder}/{filename}", f"{ndnmol_dir}/{folder}/{filename}")
        for filename in random_files:
            # moves randomly selected files to test set
            shutil.move(f"{ndnmol_dir}/{folder}/{filename}", f"{ndnmol_test_dir}/{folder}/{filename}")
    logging.info("Train validation split done!")
