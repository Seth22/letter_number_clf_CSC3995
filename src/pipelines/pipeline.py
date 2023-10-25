from src.model_metrics import metrics
from src.model_metrics import metric_calc
from src.preprocess import exception, data_retrieval, train_val_split, validation
from src.model import seq_nn_clf
import logging
import os


def make_model(input_dir, output_dir, filetype=".jpeg", train_batch_size=40, val_batch_size=40,
               loss='categorical_crossentropy', optimizer='rmsprop', metric=['accuracy'], epochs=25, steps_per_epoch=20,
               validation_steps=5, model_name="clf"):

    """
    Function:
        Pipeline to put all submodules together to train a sequential neural network on image data
        See *TensorFlow* docs for allowable arguements on all params except input_dir, output_dir, filetype, model_name
    :param input_dir: Input Directory with image files
    :param output_dir: Output directory for sorted images, ndnmol images
    :param filetype: Filetype of original images *default:  ".jpeg"*
    :param train_batch_size: Batch size for training *default: 40*
    :param val_batch_size:  Batch size for validation *default: 40*
    :param loss: loss function for seq NN *default: "cateforical_crossentropy*
    :param optimizer: Optimizer for Seq NN*default: 'accuracy'
    :param metric: Metric to determine model accuracy *default: ["accuracy"]
    :param epochs: Number of epochs for model training *default 25*
    :param steps_per_epoch: Training steps for epoch *default: 20*
    :param validation_steps: Validation steps for epoch *default 20*
    :param model_name: Name of model to save can be any string *default clf*
    :return: Logging messages
    """
    # Process and get data ready for model
    logging.info("Processing data....")

    os.mkdir(f"{output_dir}/sorted_images")
    os.mkdir(f"{output_dir}/train_val_split")
    data_retrieval.sort_images(input_dir, f"{output_dir}/sorted_images", filetype)
    # Validating sorted images
    logging.info("Validating sorted images")

    if (
    validation.val_file_number(f"{output_dir}/sorted_images", 100) or
    validation.val_filetypes(f"{output_dir}/sorted_images", ".png") or
    validation.val_file_location(f"{output_dir}/sorted_images")
    ):
        raise exception.val_error("Validation error on sorted images check logs")

    #splitting data into train and val set
    train_val_split.train_val_split(f"{output_dir}/sorted_images", f"{output_dir}/train_val_split")

    # Validating train val split
    logging.info("Validating training data")

    if (
    validation.val_file_number(f"{output_dir}/train_val_split/ndnmol", 80) or
    validation.val_filetypes(f"{output_dir}/train_val_split/ndnmol", ".png") or
    validation.val_file_location(f"{output_dir}/train_val_split/ndnmol")
    ):
        raise exception.val_error("Validation error training data set logs")

    # Validating validation data
    logging.info("Validating Validation data")

    if (
    validation.val_file_number(f"{output_dir}/train_val_split/ndnmol_test_set", 20) or
    validation.val_filetypes(f"{output_dir}/train_val_split/ndnmol", ".png") or
    validation.val_file_location(f"{output_dir}/train_val_split/ndnmol")
    ):
        raise exception.val_error("Validation error on validation set check logs")


    logging.info("Processing Done")

    # generate model data
    logging.info("Generating model data....")

    seq_nn_clf.train_datagen(f"{output_dir}/train_val_split/ndnmol", train_batch_size)
    seq_nn_clf.val_datagen(f"{output_dir}/train_val_split/ndnmol_test_set", val_batch_size)

    logging.info("Model generation done")

    # create and train model
    logging.info("Creating and training model....")

    seq_nn_clf.create_model(loss, optimizer, metric)
    # model info is tuple with 3 elements: Model, History, model.summary()
    global model_info  # remove later
    model_info = seq_nn_clf.train_model(epochs, steps_per_epoch, validation_steps, model_name)

    logging.info("Creating and training model done")

    # shows metrics
    logging.info("Doing prediction and label calculations...")
    y_true, y_pred = metric_calc.pred_from_label(f"{output_dir}/train_val_split/ndnmol_test_set", model_info[0])
    os.mkdir(f"{output_dir}/graphs")
    logging.info("creating confusion matrix")
    metrics.create_confusion_matrix(y_true, y_pred, f"{output_dir}/graphs/fig1.png")

    logging.info("Creating epoch graph")
    metrics.create_acc_val_epoch_graph(model_info[1], f"{output_dir}/graphs/fig2.png")
