
"""

Testing for 6 true samples using CNN model.

"""
import argparse
import pandas as pd
import csv
import numpy as np
import os
import numpy as np
import tensorflow as tf
import joblib
import argparse

from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
seed = 42
np.random.seed(seed)


import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

seed = 42
np.random.seed(seed)


def import_TFSMLayer_model(path_saved_model:str):
    """
    Load TFSMLayer from SavedModel directory.
    Args:
        path_saved_model: path to SavedModel directory
    Returns:
        layer: loaded TFSMLayer instance
    """
    print("Import TFSMLayer model is processing.....\n")
    layer = TFSMLayer(path_saved_model, call_endpoint='serving_default')
    return layer


def predict_one_sample_with_layer(sample:str, layer):
    """
    Predict 1 sample data with TFSMLayer model

    Args:
        sample: path to sample csv file
        layer: loaded TFSMLayer model

    Returns:
        data_test: np.array reshaped for prediction
        y_predict: predicted labels as numpy array
    """
    data = np.loadtxt(fname=sample, delimiter=",")
    data_test_one = data.reshape(1, data.shape[0], data.shape[1], 1)  # thêm channel dim

    # Dự đoán với TFSMLayer trả về dict output
    pred_dict = layer(tf.convert_to_tensor(data_test_one, dtype=tf.float32))
    key = list(pred_dict.keys())[0]  # tự lấy key đầu tiên
    pred_probs = pred_dict[key].numpy()

    y_predict = np.argmax(pred_probs, axis=1)

    return data_test_one, y_predict


def evaluation_matrix(y_test, y_predict):
    print("Evaluation_matrix is processing.....")

    accuracy_test = accuracy_score(y_test, y_predict)
    print(f"Accuracy test CNN: {accuracy_test*100:.2f}%")
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_predict, average='macro')
    print("Precision CNN:", precision)
    print("Recall CNN:", recall)
    print("F1-score CNN:", fscore, "\n")


def testcase_one_sample(path_true_sample, layer):
    print("Test_1_sample is processing......")

    data, y_predict = predict_one_sample_with_layer(str(path_true_sample), layer)

    # Lấy xác suất dự đoán:
    pred_dict = layer(tf.convert_to_tensor(data, dtype=tf.float32))
    key = list(pred_dict.keys())[0]
    y_predict_test_proba = pred_dict[key].numpy()

    labels_6_true = ["corona", "floating","noise",  "surface", "turn to turn", "void"]
    pro_6_labels = ["proba_" + i for i in labels_6_true]

    df_proba_samples = pd.DataFrame(y_predict_test_proba, columns=pro_6_labels)
    # df_6_true = pd.DataFrame([path_true_sample], columns=["Path"])
    
    df_proba_a_path = pd.concat([df_proba_samples], axis=1)
    print("\n", path_true_sample)
    print("-->Sample is predicted:", labels_6_true[y_predict[0]])
    print("\Probabilities of each label in this TESTCASE\n", df_proba_a_path)
    return df_proba_a_path


def main():
    path_saved_model = "SaveModel_CNN/my_saved_model"  # Đường dẫn tới thư mục SavedModel của bạn
    layer = import_TFSMLayer_model(path_saved_model)

    #Enter path of each sample in each testcase
    parser = argparse.ArgumentParser(description='Enter the a testing file path.')
    parser.add_argument('--path_true_sample', type=str, required=True, help='a testing file path')
    args = parser.parse_args()
    path_true_sample = args.path_true_sample
    testcase_one_sample(path_true_sample, layer)


if __name__ == "__main__":
    main()


