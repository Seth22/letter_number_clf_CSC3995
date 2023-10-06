from preprocess import data_retrieval,train_val_split, validation
from model import model

def make_model(input_dir, output_dir, filetype, train_batch_size, val_batch_size, loss, optimizer, metric, epochs, steps_per_epoch, validation_steps):
    #Process and get data ready for model\
    print("Processing data....")
    process_data(input_dir,output_dir, filetype)
    print("Done \nGenerating model data....")
    
    #generate model data
    seq_nn_datagen(output_dir, train_batch_size, val_batch_size)
    print("Done \n Creating and training model....")

    #create and train model
    make_seq_nn_model(loss, optimizer, metric, epochs, steps_per_epoch, validation_steps)
    print("Done")
#pipeline to process data
def process_data(input_dir, output_dir,filetype):
    data_retrieval.sort_images(input_dir, output_dir, filetype)
    train_val_split.train_valsplit(output_dir)

#Pipeline for Data generation for Sequential Neural Network
def seq_nn_datagen(input_dir, train_batch_size, val_batch_size):
    model.train_datagen(input_dir, train_batch_size)
    model.val_datagen(input_dir, val_batch_size)

#Pipeline to train Sequential Neural Network
def make_seq_nn_model(loss, optimizer, metric, epochs, steps_per_epoch, validation_steps):
    model.create_model(loss, optimizer, metric)
    model.train_model(epochs, steps_per_epoch, validation_steps)
