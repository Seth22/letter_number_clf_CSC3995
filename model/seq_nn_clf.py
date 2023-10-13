import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from preprocess import exception

def train_datagen(input_dir, batch_size_number):

    training_datagen = ImageDataGenerator(
        rescale=1. / 255)


    global train_generator
    train_generator = training_datagen.flow_from_directory(
        input_dir,
        target_size=(150, 150),
        class_mode='categorical',
        batch_size=batch_size_number  # Default 40 from POC
    )


def val_datagen(input_dir, input_batch_size):
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    global validation_generator
    validation_generator = validation_datagen.flow_from_directory(
        input_dir,
        target_size=(150, 150),
        class_mode='categorical',
        batch_size=input_batch_size  # default 40 from POC
    )


# default loss, optimizer and metrics from POC
def create_model(loss_input, optimizer_input, metrics_input):
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
    if train_generator is None:
        raise exception.datagen_not_found("No training datagen found, try using train_datagen function")

    if validation_generator is None:
        raise exception.datagen_not_found("No validation datagen found, try using validation_datagen function")


    history = model.fit(train_generator, epochs=epochs_number, steps_per_epoch=steps_per_epoch_number,
                        validation_data=validation_generator, verbose=verbose_input,
                        validation_steps=validation_steps_number)
    model.save(model_name)
    return model, history, model.summary()
