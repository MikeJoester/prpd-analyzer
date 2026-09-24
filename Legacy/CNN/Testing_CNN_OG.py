"""
Testing for D8 with CNN model.
"""

import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model\

import pandas as pd


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def read_testing_set(test_path:str, label_encoder):
    """
    Args:
        test_path: file CSV includes paths of training set

    Returns:
        X_train: Paths of training set
        X_test: Paths of testing set
        y_train: labels of training set already transfered to numeric by label_ encoder
        y_test: labels of testing set already transfered to numeric by label_ encoder
    """
    # logger.info("Read paths of training and testing set is processing.....\n")

    df_paths_labels_test = pd.read_csv(test_path)
    x_test = (df_paths_labels_test["Paths"]).values
    y_test = (df_paths_labels_test["True labels"]).values


    y_test = label_encoder.transform(y_test)
    indices = [0, 1, 2, 3, 4, 5]
    for i in indices:
        print(f"Label {i} according to:", label_encoder.classes_[i])
    return x_test, y_test


def loading_cnn_model(path: str):
    """
    Args:
        path: path of CNN model already saved

    Returns:
        cnn_trained: CNN model already trained
    """
    cnn_trained = load_model(path)
    return cnn_trained

def from_file_X(X_paths: list, cnn_trained: object):
    """

    Args:
        X_paths: Paths of testing data set
        cnn_trained: CNN model already trained

    Returns:
        y_predict: Labels predicted by CNN model of testing data according to paths

    """
    data = [np.loadtxt(fname=f, delimiter=",") for f in X_paths]
    data = [f.reshape(1, f.shape[0], f.shape[1]) for f in data]
    data = np.concatenate(data, axis=0)
    # data = data[..., np.newaxis]
    y_predict = np.argmax(cnn_trained.predict(data, verbose=None), axis=1)

    return y_predict
    
def predict(path: list, cnn_trained: object):
    """

    Args:
        path: Paths testing set
        cnn_trained: CNN model already trained

    Returns:
        y_predict_all: All Labels predicted by CNN model of testing data

    """
    print("Loading data complete. Starting process files ...")
    y_predict_all = []
    tmp_paths = []
    for i in path:
        tmp_paths.append(i)
        if len(tmp_paths) % 1000 == 0:
            y_predict = from_file_X(tmp_paths, cnn_trained)
            y_predict_all.extend(y_predict)
            print(f"Proceed {len(y_predict_all)} files ...")
            tmp_paths.clear()

    if tmp_paths:
        y_predict = from_file_X(tmp_paths, cnn_trained)
        y_predict_all.extend(y_predict)
        print(f"Proceed {len(y_predict_all)} files ...")
        tmp_paths.clear()

    print("Process files done. Start evaluating model performance ...")
    return y_predict_all



def print_confusion_matrix(y_numeric, y_predict, label_encoder):
    """
    Args:
        y_numeric: Labels of test set after transfer to numeric
        y_predict:Samples of testset predicted by CNN model
    """
    class_names = label_encoder.classes_
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_numeric, y_predict, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    print(cm)


def evaluation_metrics(y_numeric, y_predict):
    """
    Args:
        y_numeric: Labels of test set after transfer to numeric
        y_predict:Samples of testset predicted by CNN model
    """
    print("\n")
    print("Evaluating model performances ...")
    precision, recall, fscore, support = precision_recall_fscore_support(y_numeric, y_predict, average='macro')
    print("Precision CNN:", precision)
    print("Recall CNN:", recall)
    print("f1_score CNN:", fscore)
    print("Testing Accuracy CNN:", accuracy_score(y_numeric, y_predict) * 100)

from keras.layers import TFSMLayer


    # model_path = "SaveModel_CNN/my_saved_model.keras"


    
    
def predict_with_TFSMLayer(layer, data_test_tf, batch_size=64):
    """
    Args:
        layer: TFSMLayer loaded from SavedModel
        data_test_tf: tf.Tensor of shape (N, 3600, 128, 1)
        batch_size: batch size for inference (default=64)
    
    Returns:
        y_predict: np.array of predicted labels (int)
    """
    y_pred_probs_list = []

    for i in range(0, data_test_tf.shape[0], batch_size):
        batch = data_test_tf[i:i+batch_size]
        
        # Lấy dict output
        batch_pred_dict = layer(batch)
        key = list(batch_pred_dict.keys())[0]    # Tự động lấy key
        batch_pred = batch_pred_dict[key]

        y_pred_probs_list.append(batch_pred.numpy())

        print(f"Processed {min(i + batch_size, data_test_tf.shape[0])}/{data_test_tf.shape[0]} samples ...")

    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)
    y_predict = np.argmax(y_pred_probs, axis=1)

    print("✅ Prediction done.")
    return y_predict

def main():

    enc_path   = 'SaveModel_SVM/label_encoder.joblib'
    label_encoder = joblib.load(enc_path)
        # Load paths:
    X_test_paths, y_test = read_testing_set("Data/Paths_and_Labels_of_TestingSet.csv", label_encoder)
    
    # Load TFSMLayer:
    layer = TFSMLayer("SaveModel_CNN/my_saved_model", call_endpoint='serving_default')
    # Load data từ X_test_paths:
    data_test = []
    for path in X_test_paths:
        arr = np.loadtxt(fname=path, delimiter=",")
        arr = arr.reshape(1, arr.shape[0], arr.shape[1])
        data_test.append(arr)
    
    data_test = np.concatenate(data_test, axis=0)
    data_test = data_test[..., np.newaxis]
    
    # Convert to TensorFlow tensor:
    import tensorflow as tf
    data_test_tf = tf.convert_to_tensor(data_test, dtype=tf.float32)
    
    # Predict using function:
    y_predict_test = predict_with_TFSMLayer(layer, data_test_tf, batch_size=64)
    
    # Evaluate:
    # class_names = ["Corona","Floating", "Noise", "Surface", "Turn_to_Turn", "Void"]
    print_confusion_matrix(y_test, y_predict_test, label_encoder)
    evaluation_metrics(y_test, y_predict_test)
    


if __name__ == "__main__":
    main()
