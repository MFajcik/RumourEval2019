"""
This code contains several versions of objective functions to be used together
with parameter search functions
"""
import os

import numpy as np
from hyperopt import STATUS_OK
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score

# %%
from official_baseline.LSTM_models import LSTM_model_stance, LSTM_model_veracity
from official_baseline.branch2treelabels import branch2treelabels


def objective_function_stance_branchLSTM_RumEv(params):
    x_train = np.load(os.path.join('data_preprocessing/saved_data_RumEval2019_baseline',
                                   'train/train_array.npy'))
    y_train = np.load(os.path.join('data_preprocessing/saved_data_RumEval2019_baseline',
                                   'train/fold_stance_labels.npy'))
    y_train_cat = []
    for i in range(len(y_train)):
        y_train_cat.append(to_categorical(y_train[i], num_classes=4))
    y_train_cat = np.asarray(y_train_cat)
    x_test = np.load(os.path.join('data_preprocessing/saved_data_RumEval2019_baseline',
                                  'dev/train_array.npy'))
    y_test = np.load(os.path.join('data_preprocessing/saved_data_RumEval2019_baseline',
                                  'dev/fold_stance_labels.npy'))

    ids_test = np.load(os.path.join('data_preprocessing/saved_data_RumEval2019_baseline',
                                    'dev/tweet_ids.npy'))

    y_pred, confidence = LSTM_model_stance(x_train, y_train_cat,
                                           x_test, params)
    fids_test = []
    for i in ids_test:
        fids_test.extend(i)
    fy_pred = y_pred.flatten()
    fy_test = y_test.flatten()
    uniqtwid, uindices2 = np.unique(fids_test, return_index=True)
    uniqtwid = uniqtwid.tolist()
    uindices2 = uindices2.tolist()
    uniq_dev_prediction = [fy_pred[i] for i in uindices2]
    uniq_dev_label = [fy_test[i] for i in uindices2]

    mactest_F = f1_score(uniq_dev_prediction, uniq_dev_label, average='macro')
    acc = sum([1 for x, y in zip(uniq_dev_prediction, uniq_dev_label) if x == y]) / len(uniq_dev_label)

    print(f"validation F1 is: {mactest_F:3f} and ACC is {acc}")
    output = {'loss': 1 - mactest_F,
              'Params': params,
              'status': STATUS_OK,
              'attachments': {'ID': uniqtwid,
                              'Predictions': uniq_dev_prediction,
                              'Labels': uniq_dev_label}}
    return output


# %%

def objective_function_veracity_branchLSTM_RumEv(params):
    x_train = np.load(os.path.join('data_preprocessing/saved_dataRumEval2019',
                                   'train/train_array.npy'))
    y_train = np.load(os.path.join('data_preprocessing/saved_dataRumEval2019',
                                   'train/labels.npy'))
    y_train = to_categorical(y_train, num_classes=None)
    x_test = np.load(os.path.join('data_preprocessing/saved_dataRumEval2019',
                                  'dev/train_array.npy'))
    y_test = np.load(os.path.join('data_preprocessing/saved_dataRumEval2019',
                                  'dev/labels.npy'))
    ids_test = np.load(os.path.join('data_preprocessing/saved_dataRumEval2019',
                                    'dev/ids.npy'))
    y_pred, confidence = LSTM_model_veracity(x_train, y_train, x_test, params)
    trees, tree_prediction, tree_label, _ = branch2treelabels(ids_test,
                                                              y_test,
                                                              y_pred,
                                                              confidence)
    mactest_F = f1_score(tree_label, tree_prediction, average='macro')
    output = {'loss': 1 - mactest_F,
              'Params': params,
              'status': STATUS_OK,
              'attachments': {'ID': trees, 'Predictions': tree_prediction, 'Labels': tree_label}}
    return output
