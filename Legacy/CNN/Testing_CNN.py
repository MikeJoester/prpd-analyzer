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
from tensorflow.keras.models import load_model

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
        y_predict: Samples of testset predicted by CNN model
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
        y_predict: Samples of testset predicted by CNN model
    """
    print("\n")
    print("Evaluating model performances ...")
    precision, recall, fscore, support = precision_recall_fscore_support(y_numeric, y_predict, average='macro')
    print("Precision CNN:", precision)
    print("Recall CNN:", recall)
    print("f1_score CNN:", fscore)
    print("Testing Accuracy CNN:", accuracy_score(y_numeric, y_predict) * 100)


def main():
    enc_path = 'SaveModel_SVM/label_encoder.joblib'
    label_encoder = joblib.load(enc_path)
    
    # Load paths:
    X_test_paths, y_test = read_testing_set("Data/Paths_and_Labels_of_TestingSet.csv", label_encoder)
    
    # Load CNN model:
    cnn_model = loading_cnn_model("SaveModel_CNN/my_saved_model.keras")
    
    # Predict:
    y_predict_test = predict(X_test_paths, cnn_model)
    
    # Evaluate:
    print_confusion_matrix(y_test, y_predict_test, label_encoder)
    evaluation_metrics(y_test, y_predict_test)


if __name__ == "__main__":
    main()
