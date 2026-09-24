"""
Testing for 6 true samples by SVM model
"""
# IMPORT LIBRARY
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import pandas as pd
import numpy as np
import argparse
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
seed = 42
np.random.seed(seed)


def loading_svm_model(file_SVM_saved:object):
    """
    Args:
        file_SVM_saved: File includes SVM model saved in SVM training process

    Returns:
        svm_model: SVM model already saved
    """
    print("Loading_svm_model is processing.....\n")

    import joblib
    svm_model = joblib.load('SaveModel_SVM/svm_model.joblib')

    return svm_model


def predict_one_sample(sample:str, svm_model:object):
    """
    Args:
        sample: 1 sample each time
        svm_model: SVM model already saved

    Returns:
        extract_data_test
        y_predict_test: y_predict_test after add one sample....
    """
    data = np.loadtxt(
        fname = sample,
        delimiter = ","
        )  
    #Feature extraction
    max_each_columns = np.max(data, axis=0)
    max_devide_nb_zeros = max_each_columns / (np.count_nonzero(data, axis=0) + 1e-15)
    max_nonzeros_D3 = np.concatenate(([max_each_columns], [max_devide_nb_zeros]), axis = 0)
    extract_data = max_nonzeros_D3.reshape(1, -1)
    #Make a prediction
    y_predict = svm_model.predict(extract_data)

    return extract_data, y_predict


def testcase_one_sample(path_true_sample, svm_model):
    """
    Args:
        path_true_sample: Paths of one true sample according to one test case
        svm_model: SVM model already saved

    Returns:
        df_proba_one_path: Data frame includes Probabilities of one labels in one testcase with accorded one path
    """
    print("Test_a_sample is processing......")

    data, y_predict = predict_one_sample(str(path_true_sample), svm_model)
    y_predict_test_proba = svm_model.predict_proba(data)
    labels_5_true = ["corona", "floating","noise",  "surface", "turn to turn", "void"]
    pro_5_labels = ["proba_" + i for i in labels_5_true]

    df_proba_samples = pd.DataFrame(y_predict_test_proba, columns=pro_5_labels)
    
    df_5_true = pd.DataFrame([path_true_sample])
    df_proba_samples = df_proba_samples.reset_index(drop=True)
    df_proba_path = pd.concat([df_5_true, df_proba_samples], axis=1)
    print(path_true_sample)
    print("-->Sample is predicted:", labels_5_true[y_predict[0]])
    print("Probability of each labels in each TESTCASE\n", df_proba_samples)



def main():

    path_svm_model = "SaveModel_SVM/svm_model.joblib"
    # path_6_true_sample = ['D6_data/240222_210546[0].csv', 'D3_data/corona/230804_081747[0].csv',
    #                       'D3_data/floating/230803_135844[0].csv', 'D3_data/noise/230803_121732[0].csv',
    #                       'D3_data/surface/230803_114906[0].csv', 'D3_data/void/230804_101757[0].csv']

    parser = argparse.ArgumentParser(description='Enter the a testing file path.')
    parser.add_argument('--path_true_sample', type=str, required=True, help='a testing file path')
    args = parser.parse_args()
    path_true_sample = args.path_true_sample

    svm_model = loading_svm_model(path_svm_model)
    testcase_one_sample(path_true_sample, svm_model)




if __name__ == "__main__":
    main()

