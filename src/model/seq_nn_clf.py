import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from src.preprocess import exception


def train_datagen(input_dir, batch_size):
    """
    Function:
        Generates Training data for model(reformatting images for model) and creates a **global variable train_generator**
        this variable has training data set for model.
    Current use:
        Used in **train_model** as data to train model
    Important Notes:
        **ORDER MATTERS** This function must be called **before** train_model
    :param input_dir: Directory with training set
    :param batch_size: Batch size for training data generation
    :return: None
    """

    training_datagen = ImageDataGenerator(
        rescale=1. / 255)

    global train_generator
    train_generator = training_datagen.flow_from_directory(
        input_dir,
        target_size=(150, 150),
        class_mode='categorical',
        batch_size=batch_size
    )


def val_datagen(input_dir, batch_size):
    """
    Function:
        Generates validation data for model(reformatting images for model) and creates a **global variable
        validation_generator** this variable has validation data set for model.
    Current use:
        Used in **train_model** as data to validate model at each epoch
    Important Notes:
        **ORDER MATTERS** This function must be called **before** train_model
    :param input_dir: Directory with validation data set
    :param batch_size: Batch size for validation data generation
    :return: None
    """
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    global validation_generator
    validation_generator = validation_datagen.flow_from_directory(
        input_dir,
        target_size=(150, 150),
        class_mode='categorical',
        batch_size=batch_size  # default 40 from POC
    )


# default loss, optimizer and metrics from POC
def create_model(loss_input, optimizer_input, metrics_input):
    """
    Function:
        This function creates the model given a loss_input, optimizer_input and metrics_input(see pipeline for defaults)
        creates a Sequential Neural Network
    Current Use:
        Used in **train_model** as the main model creation function
        **Creates global variable Model** which is called upon in other functions such as train_model
    IMPORTANT NOTES:
        **ORDER MATTERS** This function must be called **before** train_model
    :param loss_input: Defined in TensorFlow model gen
    :param optimizer_input: Defined in TensorFlow model gen
    :param metrics_input: Defined in TensorFlow model gen
    :return: None

    """
    global model
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # PS: updated to 10 from 3 because we have 10 clsses and not 3
    ])
    model.compile(loss=loss_input, optimizer=optimizer_input, metrics=metrics_input)


def train_model(epochs_number, steps_per_epoch_number, validation_steps_number, model_name, verbose_input=1, ):
    """
    Function:
        Main function to generate model **other functions MUST be called first**, global variables are called here
        such as Model, Train_generator, validation_generator but are not created here
    Params:
        Defined by Tensorflow sequential model generation, see pipeline for decent default values
    Return Tuple:
        1. The model itself, useful for using the model after training to make predictions
        2. The history of the model, useful for creating metrics that are epoch related
        3. Model.summary(), useful for basic model information(see TensorFlow doc for more info)
    :return: A tuple with the 3 elements [model, history, model.summary()]
    :exception datagen_not_found: If no training or validtion data was found from train_datagen() or validation_datagen() functions
    :exception model_not_found
    """
    if train_generator is None:
        raise exception.datagen_not_found("No training datagen found, try using train_datagen function")

    if validation_generator is None:
        raise exception.datagen_not_found("No validation datagen found, try using validation_datagen function")

    if model is None:
        raise exception.model_not_found("No model found, try using create_model function")

    history = model.fit(train_generator, epochs=epochs_number, steps_per_epoch=steps_per_epoch_number,
                        validation_data=validation_generator, verbose=verbose_input,
                        validation_steps=validation_steps_number)
    model.save(model_name)
    return model, history, model.summary()
