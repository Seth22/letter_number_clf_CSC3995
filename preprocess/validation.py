import os, shutil
import exception

# This works for original dataset we used in the project(each row is equal to a class)
# Does not generalize for all datasets
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


def val_file_number(input_dir, filenumber, filetype=".png"):
    counter = 0
    for folder in os.listdir(input_dir):
        for filename in os.listdir(f"{input_dir}{folder}"):
            if filename.endswith(filetype):
                counter += 1
        if counter != filenumber:
            # may change in future as will stop at first unexpected number of files
            raise exception.file_number_exception(f"Expected {filenumber} files but got {counter}")
        counter = 0


def val_filetypes(input_dir, filetype=".jpeg"):
    # makes sure filetype has . if not adds it
    if filetype[0] != ".":
        filetype = filetype + "."

    for folder in os.listdir(input_dir):
        for filename in os.listdir(f"{input_dir}/{folder}"):
            if not filename.endswith(filetype):
                raise exception.file_type_exception(
                    f"Expected {filetype} but got {filetype.rsplit('.', 1)}, try using val_remove_filetypes")


def val_remove_filetypes(input_dir, filetype=".jpeg", remove=False, verbose=True):
    # makes sure filetype has . if not adds it
    if filetype[0] != ".":
        filetype = filetype + "."

    for folder in os.listdir(input_dir):
        for filename in os.listdir(f"{input_dir}/{folder}"):
            if not filename.endswith(filetype):
                if verbose:
                    print(f"remove file in folder:{folder}: {input_dir}/{folder}/{filename}")
                shutil.rmtree(f"{input_dir}/{folder}/{filename}")


def val_file_location(input_dir):
    for folder in os.listdir(input_dir):
        expected_filenumber_range = org_folder_number_key.get(folder)
        for filename in os.listdir(f"{input_dir}{folder}"):
            if filename.endswith(".png"):
                filename_number = int(
                    filename.split('.')[0])  # keeps only parts of filename before . and turns string into an int
            # checks if file number mod 100(gets rid of 100s place) is in expected range
            if filename_number % 100 < expected_filenumber_range[0] or filename_number % 100 > \
                    expected_filenumber_range[1]:
                # some files in n directory are multiples of 100 so an extra check is needed
                if not (folder == "n" and filename_number % 100 == 0):
                    print(f"Error: Dataset contains file in wrong folder at: {input_dir}{folder}/{filename}")


def val_file_repetition(input_dir):
    for folder in os.listdir(input_dir):
        for filename in os.listdir(f"{input_dir}/{folder}"):
            if filename in os.listdir(f"{input_dir}{folder}"):
                print(
                    f"\nERROR repeated files in training and test set \nFile:{filename} in {input_dir}/{folder} and {input_dir}/{folder}")
