from preprocess import data_retrieval, train_val_split
from model import seq_nn_clf
import logging

def make_model(input_dir, output_dir, filetype=".jpeg", train_batch_size=40, val_batch_size=40,
               loss='categorical_crossentropy', optimizer='rmprop', metric=['accuracy'], epochs=25, steps_per_epoch=20,
               validation_steps=5, model_name="clf"):
    # Process and get data ready for model
    logging.info("Processing data....")
    process_data(input_dir, output_dir, filetype)
    logging.info("Processing Done")

    # generate model data
    logging.info("Generating model data....")
    seq_nn_datagen(output_dir, train_batch_size, val_batch_size)
    logging.info("Model generation done")

    # create and train model
    logging.info("Creating and training model....")
    make_seq_nn_model(loss, optimizer, metric, epochs, steps_per_epoch, validation_steps)
    logging.info("Creating and training model done")


# pipeline to process data
def process_data(input_dir, output_dir, filetype):
    data_retrieval.sort_images(input_dir, output_dir, filetype)
    train_val_split.train_val_split(output_dir)


# Pipeline for Data generation for Sequential Neural Network
def seq_nn_datagen(input_dir, train_batch_size, val_batch_size):
    seq_nn_clf.train_datagen(input_dir, train_batch_size)
    seq_nn_clf.val_datagen(input_dir, val_batch_size)


# Pipeline to train Sequential Neural Network
def make_seq_nn_model(loss, optimizer, metric, epochs, steps_per_epoch, validation_steps):
    seq_nn_clf.create_model(loss, optimizer, metric)
    seq_nn_clf.train_model(epochs, steps_per_epoch, validation_steps)
