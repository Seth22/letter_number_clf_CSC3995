import os
import logging
from PIL import Image


def sort_images(input_dir, output_dir, filetype):
    """
    Function:
        Takes a directory containing one or more **SQUARE** image files with each row being a specified letter or number
        *see src/example/org_input_images for reference* then outputs them into labeled directories based on row
    Important Note:
        Current implementation expects the rows in order to be 0,1,7,8,9,l,m,k,p,n if different labels are needed based
        on row **row_folders** must be changed
    ChatGPT:
        Code mostly written by ChatGPt see POC for prompt
    :param input_dir: Directory with input images in a 10x10 grid
    :param output_dir: Desired output destination for labeled images
    :param filetype: Type of image file
    :return: Void, creates folders and places images in output_dir
    """
    # Create the output directory if it doesn't exist

    os.makedirs(output_dir, exist_ok=True)

    # Define the folder names for each row
    row_folders = ["0", "1", "7", "8", "9", "l", "m", "k", "p", "n"]
    counter = 1
    logging.info("Sorting input image files into classes based on row...")
    for filename in os.listdir(input_dir):
        if filename.endswith(filetype):
            input_path = os.path.join(input_dir, filename)

            # Open the image using Pillow
            image = Image.open(input_path)

            # Get the dimensions of the image
            width, height = image.size

            # Calculate the size of each small square (assuming perfect square)
            square_size = width // 10

            # Create folders for each row
            for folder_name in row_folders:
                row_folder = os.path.join(output_dir, folder_name)
                os.makedirs(row_folder, exist_ok=True)

            # Divide the image into 10x10 squares and place each row in the corresponding folder
            for i in range(10):
                for j in range(10):
                    # Crop a square from the image
                    left = j * square_size
                    upper = i * square_size
                    right = left + square_size
                    lower = upper + square_size

                    square = image.crop((left, upper, right, lower))

                    # Determine the folder for this row and save the square there
                    row_index = i
                    row_folder = os.path.join(output_dir, row_folders[row_index])
                    square.save(os.path.join(row_folder, f"{counter}.png"))
                    counter += 1
    logging.info("Sorting Done!")

