from preprocess import data_retrieval, train_val_split
from model import seq_nn_clf
import logging
import os

def make_model(input_dir, output_dir, filetype=".jpeg", train_batch_size=40, val_batch_size=40,
               loss='categorical_crossentropy', optimizer='rmsprop', metric=['accuracy'], epochs=25, steps_per_epoch=20,
               validation_steps=5, model_name="clf"):

    # Process and get data ready for model
    process_data(input_dir, output_dir, filetype)

    # generate model data
    seq_nn_datagen(output_dir, train_batch_size, val_batch_size)

    # create and train model
    make_seq_nn_model(loss, optimizer, metric, epochs, steps_per_epoch, validation_steps, model_name)

# pipeline to process data
def process_data(input_dir, output_dir, filetype):
    logging.info("Processing data....")
    os.mkdir(f"{output_dir}/sorted_images")
    os.mkdir(f"{output_dir}/train_val_split")
    data_retrieval.sort_images(input_dir, f"{output_dir}/sorted_images", filetype)
    train_val_split.train_val_split(f"{output_dir}/sorted_images",f"{output_dir}/train_val_split")
    logging.info("Processing Done")

# Pipeline for Data generation for Sequential Neural Network
def seq_nn_datagen(input_dir, train_batch_size, val_batch_size):
    logging.info("Generating model data....")
    seq_nn_clf.train_datagen(f"{input_dir}/train_val_split/ndnmol", train_batch_size)
    seq_nn_clf.val_datagen(f"{input_dir}/train_val_split/ndnmol_test_set", val_batch_size)
    logging.info("Model generation done")


# Pipeline to train Sequential Neural Network
def make_seq_nn_model(loss, optimizer, metric, epochs, steps_per_epoch, validation_steps, model_name):
    logging.info("Creating and training model....")
    seq_nn_clf.create_model(loss, optimizer, metric)
    seq_nn_clf.train_model(epochs, steps_per_epoch, validation_steps,model_name)
    logging.info("Creating and training model done")

