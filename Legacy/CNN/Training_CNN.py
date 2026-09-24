"""
Training CNN model from the data train
"""
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from datetime import datetime
import keras
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

# tf.config.experimental.reset_memory_stats('GPU:0')

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def read_training_and_testing_set(train_path:str, test_path:str, label_encoder):
    """
    Args:
        train_path:file CSV includes paths of training set
        test_path: file CSV includes paths of training set

    Returns:
        X_train: Paths of training set
        X_test: Paths of testing set
        y_train: labels of training set already transfered to numeric by label_ encoder
        y_test: labels of testing set already transfered to numeric by label_ encoder
    """
    logger.info("Read paths of training and testing set is processing.....\n")

    df_paths_labels_train = pd.read_csv(train_path)
    x_train = (df_paths_labels_train["Paths"]).values
    y_train = (df_paths_labels_train["True labels"]).values


    df_paths_labels_test = pd.read_csv(test_path)
    x_test = (df_paths_labels_test["Paths"]).values
    y_test = (df_paths_labels_test["True labels"]).values

    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)
    return x_train, x_test, y_train, y_test


def import_one_sample(sample:str, data_set):
    """
    Args:
        sample: 1 sample
        data_set:data of one to 6 samples

    Returns:
        data_test:data_set:data of one to 6 samples
    """
    data = np.loadtxt(
        fname=sample,
        delimiter=","
    )
    data = data.reshape(1, data.shape[0], data.shape[1])
    data_set.append(data)
    return data_set


def import_data_set(X):
    """
    Args:
        X: paths of train data or test data that you want to import

    Returns:
        data: train data or test data that you want to import 
    """
    data_set = []
    for sample in X:
        data_set = import_one_sample(sample, data_set)

    data_set = np.concatenate(data_set, axis=0)
    print("data_set", data_set.shape)
    return data_set


def define_architecture_cnn_model(out_shape):
    """
    Args:
        out_shape: Shape of output of CNN_model

    Returns:
        model : Architecture of CNN model

    """
    model = keras.Sequential(
        [   keras.Input(shape = (3600, 128,1)),
            keras.layers.Conv2D(8, 3, 1, padding='same', activation='relu', name='conv2d_1'),
            keras.layers.MaxPooling2D((3, 1), name='max_pool_1'),

            keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv2d_2'),
            keras.layers.MaxPooling2D((3, 1), name='max_pool_2'),

            keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv2d_4'),
            keras.layers.MaxPooling2D((2, 2), name='max_pool_3'),

            keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv2d_5'),
            keras.layers.MaxPooling2D((2, 2), name='max_pool_4'),

            keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv2d_6'),
            keras.layers.MaxPooling2D((2, 2), name='max_pool_5'),

            keras.layers.Flatten(name='Flatten_1'),
            keras.layers.Dropout(0.2, seed=42),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dropout(0.2, seed=42),
            keras.layers.Dense(out_shape, activation='softmax', name='dense_3')
        ]
    )
    return model




def build_cnn_model(data_train, y_train, data_test, y_test, batch_size, epochs, output_shape):
    """
    Args:
        data_train: The training set that is used to train CNN model
        y_train: The labels of training set that is used to train CNN model
        data_test: The validation set that is used to validate CNN model
        y_test:
        batch_size:
        epochs:
        input_shape: Shape of input of CNN_model
        output_shape: Shape of output of CNN_model

    Returns:
        model: CNN_model after training
    """
    logger.info("Build cnn model is processing.....")

    model = define_architecture_cnn_model(output_shape)
    # checkpointer = save_the_best_epoch()

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'], )
    train_model = model.fit(data_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_data=(data_test, y_test),
                            verbose="auto",
                            )
    model.export("SaveModel_CNN/my_saved_model")
    model.save("SaveModel_CNN/my_saved_model.keras")  # Save as Keras format for feature extraction
    print("built")
    return train_model

def plot_accuracy(history):
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Testing Accuracy')
    plt.title('Training vs Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():

    enc_path   = 'SaveModel_SVM/label_encoder.joblib'
    label_encoder = joblib.load(enc_path)


    x_train, x_test, y_train, y_test = read_training_and_testing_set("Data/Paths_and_Labels_of_TrainingSet.csv",
                                                                     "Data/Paths_and_Labels_of_TestingSet.csv", label_encoder)


    
    logger.info("Import data of training set is processing....")
    data_train = import_data_set(x_train)
    logger.info("Import data of  set is processing....\n")
    data_test = import_data_set(x_test)
    data_train = data_train[..., np.newaxis]
    data_test = data_test[..., np.newaxis]
    history = build_cnn_model(data_train, y_train, data_test, y_test, batch_size=64, epochs=100, output_shape=6)


    # Accuracy testing theo từng epoch
    # Accuracy train theo từng epoch
    train_acc = history.history['accuracy']
    
    # Accuracy validation theo từng epoch
    val_acc = history.history['val_accuracy']
    plot_accuracy(history)
    # print("Train acc từng epoch:", train_acc)
    # print("Val acc từng epoch:", val_acc)
    last_epoch = len(history.history['accuracy'])
    last_train_acc = history.history['accuracy'][-1]
    last_val_acc = history.history['val_accuracy'][-1]
    print(f"Accuracy at last epoch ({last_epoch}): Training accuracy: {last_train_acc*100:.4f}, val = {last_val_acc:.4f}")

if __name__ == "__main__":
    main()