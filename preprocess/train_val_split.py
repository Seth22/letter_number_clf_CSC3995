import os, shutil
import random
import data_retrieval, exception


# splits class sorted images into a training and validation split
# if no input_dir used, uses same dir as used in data_retrieval
def train_val_split(output_dir, input_dir=data_retrieval.sorted_images_dir):
    class_dirs = ['0', '1', '7', '8', '9', 'k', 'l', 'm', 'n', 'p']

    # create target directories ndnmol and ndnmol_test_set and set variables to remember location

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
