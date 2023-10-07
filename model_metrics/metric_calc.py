import os
import random
import shutil
import numpy as np

from keras.preprocessing import image


# used to select random files from a dataset for later graphs like confusion matrix
def create_select_pred_files(input_dir, output_dir, sample=10):
    os.mkdir(output_dir)
    for folder in os.listdir(input_dir):
        random_files = random.sample(os.listdir(f"{output_dir}/{folder}"), sample)
        os.mkdir(f"{output_dir}/{folder}")
        for filename in random_files:
            shutil.copy(f"{output_dir}/{folder}/{filename}", f"{input_dir}/{folder}/{filename}")


pred_number = {
    '0': 0,
    '1': 1,
    "7": 2,
    "8": 3,
    "9": 4,
    "k": 5,
    "l": 6,
    "m": 7,
    "n": 8,
    "p": 9,
}

def pred_from_label(input_dir, output_dir, model):
    y_true = [None] * 100
    y_pred = [None] * 100

    i = 0
    for folder in os.listdir(input_dir):
        for filename in os.listdir(f"{input_dir}/{folder}"):
            img = image.load_img(f"{input_dir}/{folder}/{filename}", target_size=(150, 150))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            y_pred[i] = model.predict(images, verbose="False")
            y_true[i] = pred_number.get(folder)
            i += 1
    return y_true, y_pred