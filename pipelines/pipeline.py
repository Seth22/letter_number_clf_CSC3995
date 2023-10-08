from model_metrics import metric_calc, metrics
from preprocess import data_retrieval, train_val_split, validation
from model import seq_nn_clf
import logging
import os

def make_model(input_dir, output_dir, filetype=".jpeg", train_batch_size=40, val_batch_size=40,
               loss='categorical_crossentropy', optimizer='rmsprop', metric=['accuracy'], epochs=25, steps_per_epoch=20,
               validation_steps=5, model_name="clf"):

    # Process and get data ready for model
    logging.info("Processing data....")

    os.mkdir(f"{output_dir}/sorted_images")
    os.mkdir(f"{output_dir}/train_val_split")
    data_retrieval.sort_images(input_dir, f"{output_dir}/sorted_images", filetype)
    train_val_split.train_val_split(f"{output_dir}/sorted_images", f"{output_dir}/train_val_split")

    logging.info("Processing Done")

    # generate model data
    logging.info("Generating model data....")

    seq_nn_clf.train_datagen(f"{output_dir}/train_val_split/ndnmol", train_batch_size)
    seq_nn_clf.val_datagen(f"{output_dir}/train_val_split/ndnmol_test_set", val_batch_size)

    logging.info("Model generation done")

    # create and train model
    logging.info("Creating and training model....")

    seq_nn_clf.create_model(loss, optimizer, metric)
    #model info is tuple with 3 elements: Model, History, model.summary()
    global model_info #remove later
    model_info = seq_nn_clf.train_model(epochs, steps_per_epoch, validation_steps,model_name)

    logging.info("Creating and training model done")

    #shows metrics
    logging.info("Doing prediction and label calculations...")
    y_true, y_pred = metric_calc.pred_from_label(f"{output_dir}/train_val_split/ndnmol_test_set", model_info[0])

    logging.info("creating confusion matrix")
    metrics.create_confusion_matrix(y_true, y_pred)

    logging.info("Creating epoch graph")
    metrics.create_acc_val_epoch_graph(model_info[2])
