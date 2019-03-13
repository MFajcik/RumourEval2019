# This script easily replicates ensemble results from the paper

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from utils.utils import setup_logging

# These predictions are included in TOP-N ensemble
# it was found via method find_best_ensemble_greedy from ensembling/ensemble_helper.py
TOP_N_ensemble = [
    "result_F1_0.57948_L_0.6698856112670224_2019-01-28_08:24_pcknot5.npy",
    "result_F1_0.57759_L_0.703442574330578_2019-01-28_00:15_pcbirger.npy",
    "result_F1_0.57623_L_0.6621931040825227_2019-01-28_00:32_pcknot5.npy",
    "result_F1_0.57526_L_0.6638631148319039_2019-01-27_08:12_pcknot4.npy",
    "result_F1_0.57423_L_0.7102468566180802_2019-01-28_17:03_pcknot5.npy",
    "result_F1_0.57371_L_0.6669414722463592_2019-01-27_00:46_pcknot5.npy",
    "result_F1_0.56750_L_0.6898565446440823_2019-01-26_20:31_pcknot4.npy",
    "result_F1_0.56656_L_0.699664715034862_2019-01-27_15:57_pcbirger.npy",
    "result_F1_0.56460_L_0.724339671515812_2019-01-28_15:53_pcbirger.npy",
    "result_F1_0.56433_L_0.663498227135592_2019-01-28_13:27_pcknot2.npy",
    "result_F1_0.56313_L_0.689033422880176_2019-01-26_20:39_pcknot4.npy",
    "result_F1_0.56069_L_0.670826427442727_2019-01-27_02:10_pcknot4.npy",
    "result_F1_0.55930_L_0.6865916204641289_2019-01-27_16:14_pcbirger.npy",
    "result_F1_0.55580_L_0.7056901221467318_2019-01-26_20:24_pcknot4.npy",
    "result_F1_0.55509_L_0.7102856230281916_2019-01-28_00:06_pcbirger.npy",
    "result_F1_0.55504_L_0.6975949840002625_2019-01-27_23:51_pcbirger.npy",
    "result_F1_0.55092_L_0.6955123813847969_2019-01-28_12:34_pcknot4.npy"
]

# These predictions are not included in EXC-N ensemble
# it was found via method remove_worst_k from ensembling/ensemble_helper.py
EXC_N_ensemble = [
    "result_F1_0.56266_L_0.749426320401043",
    "result_F1_0.55143_L_0.712958636656001",
    "result_F1_0.55581_L_0.695809597395501",
    "result_F1_0.56191_L_0.661831214200615",
    "result_F1_0.57948_L_0.669885611267022",
    "result_F1_0.57292_L_0.705603626619452"
]


def evaluate(data="test", ensemble_type="TOP_N", strategy="avg_softmaxes"):
    path = f"predictions/numpy_final_all_{'VAL' if data == 'validation' else 'TEST'}"
    files = sorted(os.listdir(path))
    prefix = "val_" if data == 'validation' else "test_"
    valid = [f for f in files if f.startswith(prefix) and f.endswith("npy")]

    if ensemble_type == "TOP_N":
        valid_ensemble_subset = [f"{prefix}{s}" for s in list(TOP_N_ensemble)]
        result_files = [f for f in valid if "result" in f and f in valid_ensemble_subset]
    else:  # EXC_N
        def has_worst_substr(f):
            for w in EXC_N_ensemble:
                if w in f: return True
            return False

        result_files = [f for f in valid if "result" in f and not has_worst_substr(f)]
    print("Ensemble is build from following files:")
    print(result_files)
    print(f"{len(result_files)} files total")

    label_file = [f for f in valid if "labels" in f][0]
    labels = np.load(os.path.join(path, label_file))
    result_matrices = [np.load(os.path.join(path, result_file)) for result_file in
                       result_files]
    results = np.array(result_matrices)

    tweet_ids = open(f"saved/ensemble/numpy_result/{prefix}ids.txt", "r").read().split()

    if strategy == "average_logits":
        feats = np.average(results, 0)
        results = torch.Tensor(feats)
    elif strategy == "sum_softmaxes":  # summing has same effect as averaging
        results = torch.Tensor(results)
        # Models x batch x classes
        results = F.softmax(results, -1)
        results = torch.sum(results, 0)
    elif strategy == "weighted_softmax_sum":
        results = torch.Tensor(results)
        # Models x batch x classes
        results = F.softmax(results, -1)
        for k in range(results.shape[0]): results[k] = results[k] * weights[k]
        results = torch.sum(results, 0)
    elif strategy == "avg_softmaxes":
        # Models x batch x classes
        results = torch.Tensor(results)
        results = F.softmax(results, -1)
        results = torch.mean(results, 0)

    else:
        return
    labels = torch.Tensor(labels).long()

    softmaxed_results = results if strategy == "avg_softmaxes" else F.softmax(results, -1)
    maxpreds, argmaxpreds = torch.max(softmaxed_results, dim=1)

    total_preds = list(argmaxpreds.cpu().numpy())
    total_labels = list(labels.cpu().numpy())

    correct_vec = argmaxpreds == labels
    total_correct = torch.sum(correct_vec).item()

    acc = total_correct / results.shape[0]
    F1 = metrics.f1_score(total_labels, total_preds, average="macro")
    F1_cls = metrics.f1_score(total_labels, total_preds, average=None)
    return acc, F1, tuple(F1_cls)


def print_results(acc, F1, per_class_F1):
    print(f"Acc: {acc}\n F1: {F1}\n C1_F1: "
          f"{per_class_F1[0]}\n C2_F1: {per_class_F1[1]}\n C3_F1: {per_class_F1[2]}\n C4_F1: {per_class_F1[3]}")


if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath="logs/",
                  config_path="configurations/logging.yml")
    # EXC_N
    print("-"*50)
    data, ensemble_type, strategy = "validation", "EXC_N", "avg_softmaxes"
    print(f"data: {data}, ensemble type: {ensemble_type}, strategy type: {strategy}")
    results = evaluate(data, ensemble_type, strategy)
    print_results(*results)

    print("-"*50)
    data = "test"
    print(f"data: {data}, ensemble type: {ensemble_type}, strategy type: {strategy}")
    results = evaluate(data, ensemble_type, strategy)
    print_results(*results)

    # TOP_N
    print("-"*50)
    data, ensemble_type, strategy = "validation", "TOP_N", "avg_softmaxes"
    print(f"data: {data}, ensemble type: {ensemble_type}, strategy type: {strategy}")
    results = evaluate(data, ensemble_type, strategy)
    print_results(*results)

    print("-"*50)
    data = "test"
    print(f"data: {data}, ensemble type: {ensemble_type}, strategy type: {strategy}")
    results = evaluate(data, ensemble_type, strategy)
    print_results(*results)

    # TOP_N_S (BEST RESULTS ON TEST DATA)
    print("-"*50)
    data, ensemble_type, strategy = "validation", "TOP_N", "average_logits"
    print(f"data: {data}, ensemble type: {ensemble_type}, strategy type: {strategy}")
    results = evaluate(data, ensemble_type, strategy)
    print_results(*results)

    print("-"*50)
    data = "test"
    print(f"data: {data}, ensemble type: {ensemble_type}, strategy type: {strategy}")
    results = evaluate(data, ensemble_type, strategy)
    print_results(*results)
