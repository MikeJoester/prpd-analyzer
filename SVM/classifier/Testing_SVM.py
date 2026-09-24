
"""
Testing for all label in D3 and D6 dataset with SVM model.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder


def read_testing_set_path(file, label_encoder):
    """
    Args:
        file :file CSV includes paths of testing set
    Returns:
        X_test: Paths of training set
        y_test: labels of training set already transfered to numeric by label_ encoder
        label_encoder: 
    """
    #replace the "file name" of sample to the"path_of sample"
    df_paths_labels_train = pd.read_csv(file)

    # transfer to numpy array
    X_test = (df_paths_labels_train["Paths"]).values
    y_test = (df_paths_labels_train["True labels"]).values
    y_test = label_encoder.transform(y_test)
    print("Read_path_of_testing_set is processing.....\n")
    return X_test, y_test


def load_SVM_model(path: str):
    """

    Args:
        path: path of SVM model already saved

    Returns:
        svm_model: SVM model already traind

    """
    svm_model = joblib.load(path)
    return svm_model


def feature_extraction(numpy_array):
    """

    Args:
        numpy_array: raw datas

    Returns:
        extract_data: data after featuring extraction

    """
    max_each_columns = np.max(numpy_array, axis=0)
    max_devide_nb_zeros = max_each_columns / (np.count_nonzero(numpy_array, axis=0) + 1e-15)
    max_nonzeros_D3 = np.concatenate(([max_each_columns], [max_devide_nb_zeros]), axis=0)
    extract_data = max_nonzeros_D3.reshape(1, -1)
    return extract_data


def from_file_to_extra(paths: list, svm_model: object):
    """

    Args:
        paths: paths of data
        svm_model: svm model already trained

    Returns:
        y_predict: labels predicted by the SVM model

    """
    data = [np.loadtxt(fname=f, delimiter=",") for f in paths]
    data = [feature_extraction(numpy_array) for numpy_array in data]
    data = np.concatenate(data, axis=0)
    # Make a prediction
    y_predict = svm_model.predict(data)

    return y_predict


def predict(path: list, svm_model: object):
    """

    Args:
        path: all paths of testing data
        svm_model: svm model already trained

    Returns:
        y_predict_test: labels predicted by the SVM model

    """
    print("Loading data complete. Starting process files ...")
    y_predict_test = []
    tmp_paths = []
    for i in path:
        tmp_paths.append(i)
        if len(tmp_paths) % 1000 == 0:
            y_predict = from_file_to_extra(tmp_paths, svm_model)
            y_predict_test.extend(y_predict)
            print(f"Proceed {len(y_predict_test)} files ...")
            tmp_paths.clear()

    if tmp_paths:
        y_predict = from_file_to_extra(tmp_paths, svm_model)
        y_predict_test.extend(y_predict)
        print(f"Proceed {len(y_predict_test)} files ...")
        tmp_paths.clear()
    return y_predict_test


def print_confusion_matrix(y_numeric, y_predict_test, svm_model: object, label_encoder):
    """

    # Args:
    #     y_numeric: Labels of Testing data after transferring to numeric
    #     y_predict_test: All Labels predicted by SVM model of testing data
    #     svm_model: svm model already trained
    # """
    # print("svm_model.classes_", svm_model.classes_)
    class_names = label_encoder.classes_
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_numeric, y_predict_test, labels=labels)
    
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
    print(cm)


def evaluation_metrics(y_numeric, y_predict_test):
    """

    Args:
        y_numeric: Labels of D3 and D6 after transferring to numeric
        y_predict_test: All Labels predicted by SVM model of testing data

    """
    print("Process files done. Start evaluating model performance ...")
    acuratecy_test = accuracy_score(y_numeric, y_predict_test)
    print(f"Testing accuracy: SVM {acuratecy_test * 100}")
    precision, recall, fscore, support = precision_recall_fscore_support(y_numeric, y_predict_test, average='macro')
    print("Precision SVM:", precision)
    print("Recall SVM:", recall)
    print("f1_score SVM:", fscore)


def main():
    enc_path   = 'SaveModel_SVM/label_encoder.joblib'
    label_encoder = joblib.load(enc_path)


    model_path = 'SaveModel_SVM/svm_model.joblib'
    Paths_labels_test = "Data/Paths_and_Labels_of_TestingSet.csv"
    X_test, y_test= read_testing_set_path(Paths_labels_test, label_encoder)

    from collections import Counter
    print("\n===== CLASS DISTRIBUTION (TEST) =====")
    test_counts = Counter(y_test)
    for class_idx, count in test_counts.items():
        print(f"Class {label_encoder.inverse_transform([class_idx])[0]}: {count}")
    print("======================================\n")
    


    
    svm_model = load_SVM_model(model_path)
    y_predict_test = predict(X_test, svm_model)

    print_confusion_matrix(y_test, y_predict_test, svm_model, label_encoder)
    evaluation_metrics(y_test, y_predict_test)


if __name__ == "__main__":
    main()
