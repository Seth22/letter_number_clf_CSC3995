import os, shutil
import logging



def val_file_number(input_dir, filenumber, filetype=".png"):
    """
    Function:
        Given a directory with subdirectories containing image files will count how many image files are in each subdirectory
        if any subdirectory does not have specified file number returns error as true.
    Use:
        To make sure we have expected number of image files after processing original input images such as after using
        functions in data_retrieval or train_val_split
    :param input_dir: Directory with subdirectories that contain imagefiles
    :param filenumber: Expected number of files
    :param filetype: Type of image file *defaults to .png* adds '.' if not present
    :return: Error, True if any subdirectory does not have expected number of files, false otherwise
    """
    # makes sure filetype has . if not adds it
    if filetype[0] != ".":
        filetype =  "." + filetype

    error = False
    counter = 0
    for folder in os.listdir(input_dir):
        for filename in os.listdir(f"{input_dir}/{folder}"):
            if filename.endswith(filetype):
                counter += 1
        if counter != filenumber:
            logging.error(f"Expected {filenumber} files but got {counter}")
            error = True
        counter = 0
    return error

def val_filetypes(input_dir, filetype=".jpeg"):
    """
    Function:
        Goes through each subdirectory within input_dir and checks that the filetypes are just expected image files
    Use:
        To check that our input to our ML algorithm is only images not anything else
    :param input_dir: A directory with subdirectories containing image files
    :param filetype: Image file type, **defaults to .jpeg**, *if filetype does not have . will automatically add it*
    :return: Error, True if any file within any of the subdirectories is not the filetype we expect, false otherwise
    """
    error = False
    # makes sure filetype has . if not adds it
    if filetype[0] != ".":
        filetype = "." + filetype

    for folder in os.listdir(input_dir):
        for filename in os.listdir(f"{input_dir}/{folder}"):
            if not filename.endswith(filetype):
                logging.error(f"Expected {filetype} but got {filetype.rsplit('.', 1)}, try using val_remove_filetypes")
                error = True
    return error


def val_remove_filetypes(input_dir, filetype):
    """
        Function:
            Goes through each subdirectory within input_dir and checks that the filetypes are just expected image files and
            *removes* files that are not our expected image type
        Use:
            To check that our input to our ML algorithm is only images not anything else and clean out the input
        :param input_dir: A directory with subdirectories containing image files
        :param filetype: Image file type, *if filetype does not have . will automatically add it*
        :return: None
        """
    # makes sure filetype has . if not adds it
    if filetype[0] != ".":
        filetype =  "." + filetype

    for folder in os.listdir(input_dir):
        for filename in os.listdir(f"{input_dir}/{folder}"):
            if not filename.endswith(filetype):
                logging.info(f"remove file in folder:{folder}: {input_dir}/{folder}/{filename}")
                os.remove(f"{input_dir}/{folder}/{filename}")


def val_file_location(input_dir):
    """
    Function:
        Checks if files are in expected location based on row number works because preprocess/data_retrieval.sortImages()
        names files in order(0-1,000)
    Important Note:
        Will work on example dataset however does not generalize, Dictionary org_folder_number_key specific to orginal
        dataset in src/example/org_input_images
    :param input_dir: A directory with subdirectories containing sorted image files
    :return: Error, true if any file is not in expected subdirectory, false otherwise
    """

    org_folder_number_key = {
        "0": [1, 10],
        "1": [11, 20],
        "7": [21, 30],
        "8": [31, 40],
        "9": [41, 50],
        "k": [71, 80],
        "l": [51, 60],
        "m": [61, 70],
        "n": [91, 100],
        "p": [81, 90],
    }

    error = False
    for folder in os.listdir(input_dir):
        expected_filenumber_range = org_folder_number_key.get(folder)
        for filename in os.listdir(f"{input_dir}/{folder}"):
            if filename.endswith(".png"):
                filename_number = int(
                    filename.split('.')[0])  # keeps only parts of filename before . and turns string into an int
            # checks if file number mod 100(gets rid of 100s place) is in expected range
            if filename_number % 100 < expected_filenumber_range[0] or filename_number % 100 > \
                    expected_filenumber_range[1]:
                # some files in n directory are multiples of 100 so an extra check is needed
                if not (folder == "n" and filename_number % 100 == 0):
                    logging.error(f"Dataset contains file in wrong folder at: {input_dir}/{folder}/{filename}")
                    error = True
    return error

def val_file_repetition(input_dir):
    """
    Function:
        Checks if any file is repeated in the folders, essentially checks if a filename shows up twice within the subdirectories
    :param input_dir: Directory with subdirectories containing image files
    :return: Error, True if any repeated files found, False
    """
    error = False
    for folder in os.listdir(input_dir):
        for filename in os.listdir(f"{input_dir}/{folder}"):
            if filename in os.listdir(f"{input_dir}/{folder}"):
                logging.error(
                    f"Repeated files in training and test set \nFile:{filename} in {input_dir}/{folder} and {input_dir}/{folder}")
                error = True
    return error
