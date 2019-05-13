import json
import logging
import os
import random
import statistics
import sys
import numpy as np
import torch

from sklearn import metrics
from torch.nn import functional as F
from utils.utils import setup_logging, map_stance_label_to_s

worst_k = [
    "result_F1_0.56266_L_0.749426320401043",
    "result_F1_0.55143_L_0.712958636656001",
    "result_F1_0.55581_L_0.695809597395501",
    "result_F1_0.56191_L_0.661831214200615",
    "result_F1_0.57948_L_0.669885611267022",
    "result_F1_0.57292_L_0.705603626619452"
]
# 0.6235170002181254result_F1_0.56750_L_0.689856544644082
found_best_ensemble5 = ["val_result_F1_0.57623_L_0.6621931040825227_2019-01-28_00:32_pcknot5.npy",
                        "val_result_F1_0.57578_L_0.6985141360841068_2019-01-27_00:54_pcknot5.npy",
                        "val_result_F1_0.57526_L_0.6638631148319039_2019-01-27_08:12_pcknot4.npy",
                        "val_result_F1_0.56750_L_0.6898565446440823_2019-01-26_20:31_pcknot4.npy",
                        "val_result_F1_0.56656_L_0.699664715034862_2019-01-27_15:57_pcbirger.npy",
                        "val_result_F1_0.56460_L_0.724339671515812_2019-01-28_15:53_pcbirger.npy",
                        "val_result_F1_0.56315_L_0.6799039991718097_2019-01-28_16:35_pcbirger.npy",
                        "val_result_F1_0.56313_L_0.689033422880176_2019-01-26_20:39_pcknot4.npy",
                        "val_result_F1_0.56186_L_0.6923334481360296_2019-01-27_16:06_pcbirger.npy",
                        "val_result_F1_0.56069_L_0.670826427442727_2019-01-27_02:10_pcknot4.npy",
                        "val_result_F1_0.55509_L_0.7102856230281916_2019-01-28_00:06_pcbirger.npy"]

#  0.6243492230990049
found_best_ensemble4 = ["val_result_F1_0.57759_L_0.703442574330578_2019-01-28_00:15_pcbirger.npy",
                        "val_result_F1_0.57578_L_0.6985141360841068_2019-01-27_00:54_pcknot5.npy",
                        "val_result_F1_0.57423_L_0.7102468566180802_2019-01-28_17:03_pcknot5.npy",
                        "val_result_F1_0.56754_L_0.6702977421811488_2019-01-28_15:44_pcknot4.npy",
                        "val_result_F1_0.56750_L_0.6898565446440823_2019-01-26_20:31_pcknot4.npy",
                        "val_result_F1_0.56659_L_0.6476175408117534_2019-01-28_00:40_pcknot5.npy",
                        "val_result_F1_0.56583_L_0.7086059594893391_2019-01-28_16:50_pcbirger.npy",
                        "val_result_F1_0.56552_L_0.6853782425250486_2019-01-28_13:18_pcknot2.npy",
                        "val_result_F1_0.56460_L_0.724339671515812_2019-01-28_15:53_pcbirger.npy",
                        "val_result_F1_0.56313_L_0.689033422880176_2019-01-26_20:39_pcknot4.npy",
                        "val_result_F1_0.56188_L_0.6979652558757128_2019-01-28_08:08_pcknot5.npy",
                        "val_result_F1_0.56069_L_0.670826427442727_2019-01-27_02:10_pcknot4.npy",
                        "val_result_F1_0.55930_L_0.6865916204641289_2019-01-27_16:14_pcbirger.npy",
                        "val_result_F1_0.55624_L_0.7194730919406742_2019-01-26_21:55_pcknot4.npy",
                        "val_result_F1_0.55580_L_0.7056901221467318_2019-01-26_20:24_pcknot4.npy",
                        "val_result_F1_0.55509_L_0.7102856230281916_2019-01-28_00:06_pcbirger.npy"]

# 0.6257583490998202
found_best_ensemble = [
    "val_result_F1_0.57948_L_0.6698856112670224_2019-01-28_08:24_pcknot5.npy",
    "val_result_F1_0.57759_L_0.703442574330578_2019-01-28_00:15_pcbirger.npy",
    "val_result_F1_0.57623_L_0.6621931040825227_2019-01-28_00:32_pcknot5.npy",
    "val_result_F1_0.57526_L_0.6638631148319039_2019-01-27_08:12_pcknot4.npy",
    "val_result_F1_0.57423_L_0.7102468566180802_2019-01-28_17:03_pcknot5.npy",
    "val_result_F1_0.57371_L_0.6669414722463592_2019-01-27_00:46_pcknot5.npy",
    "val_result_F1_0.56750_L_0.6898565446440823_2019-01-26_20:31_pcknot4.npy",
    "val_result_F1_0.56656_L_0.699664715034862_2019-01-27_15:57_pcbirger.npy",
    "val_result_F1_0.56460_L_0.724339671515812_2019-01-28_15:53_pcbirger.npy",
    "val_result_F1_0.56433_L_0.663498227135592_2019-01-28_13:27_pcknot2.npy",
    "val_result_F1_0.56313_L_0.689033422880176_2019-01-26_20:39_pcknot4.npy",
    "val_result_F1_0.56069_L_0.670826427442727_2019-01-27_02:10_pcknot4.npy",
    "val_result_F1_0.55930_L_0.6865916204641289_2019-01-27_16:14_pcbirger.npy",
    "val_result_F1_0.55580_L_0.7056901221467318_2019-01-26_20:24_pcknot4.npy",
    "val_result_F1_0.55509_L_0.7102856230281916_2019-01-28_00:06_pcbirger.npy",
    "val_result_F1_0.55504_L_0.6975949840002625_2019-01-27_23:51_pcbirger.npy",
    "val_result_F1_0.55092_L_0.6955123813847969_2019-01-28_12:34_pcknot4.npy"
]

# 0.6240366269728898
found_best_ensemble3 = [
    "val_result_F1_0.57759_L_0.703442574330578_2019-01-28_00:15_pcbirger.npy",
    "val_result_F1_0.57623_L_0.6621931040825227_2019-01-28_00:32_pcknot5.npy",
    "val_result_F1_0.57371_L_0.6669414722463592_2019-01-27_00:46_pcknot5.npy",
    "val_result_F1_0.56750_L_0.6898565446440823_2019-01-26_20:31_pcknot4.npy",
    "val_result_F1_0.56555_L_0.7051125253749022_2019-01-27_02:47_pcknot5.npy",
    "val_result_F1_0.56532_L_0.6721716629406512_2019-01-27_00:23_pcknot5.npy",
    "val_result_F1_0.56460_L_0.724339671515812_2019-01-28_15:53_pcbirger.npy",
    "val_result_F1_0.56313_L_0.689033422880176_2019-01-26_20:39_pcknot4.npy",
    "val_result_F1_0.56191_L_0.661831214200615_2019-01-28_13:35_pcknot2.npy",
    "val_result_F1_0.56186_L_0.6923334481360296_2019-01-27_16:06_pcbirger.npy",
    "val_result_F1_0.56076_L_0.6748223714530468_2019-01-26_20:48_pcknot4.npy",
    "val_result_F1_0.56069_L_0.670826427442727_2019-01-27_02:10_pcknot4.npy",
    "val_result_F1_0.55504_L_0.6975949840002625_2019-01-27_23:51_pcbirger.npy",
    "val_result_F1_0.55055_L_0.731588928797094_2019-01-28_02:59_pcknot4.npy"
]

found_best_ensemble1 = ['val_result_F1_0.55055_L_0.731588928797094_2019-01-28_02:59_pcknot4.npy',
                        'val_result_F1_0.55504_L_0.6975949840002625_2019-01-27_23:51_pcbirger.npy',
                        'val_result_F1_0.55509_L_0.7102856230281916_2019-01-28_00:06_pcbirger.npy',
                        'val_result_F1_0.55517_L_0.6874408203495963_2019-01-27_01:13_pcknot4.npy',
                        'val_result_F1_0.55930_L_0.6865916204641289_2019-01-27_16:14_pcbirger.npy',
                        'val_result_F1_0.56076_L_0.6748223714530468_2019-01-26_20:48_pcknot4.npy',
                        'val_result_F1_0.56245_L_0.6623249796954689_2019-01-27_00:31_pcknot5.npy',
                        'val_result_F1_0.56256_L_0.7096827939514201_2019-01-26_19:03_pcbirger.npy',
                        'val_result_F1_0.56313_L_0.689033422880176_2019-01-26_20:39_pcknot4.npy',
                        'val_result_F1_0.56460_L_0.724339671515812_2019-01-28_15:53_pcbirger.npy',
                        'val_result_F1_0.56552_L_0.6853782425250486_2019-01-28_13:18_pcknot2.npy',
                        'val_result_F1_0.56750_L_0.6898565446440823_2019-01-26_20:31_pcknot4.npy',
                        'val_result_F1_0.57094_L_0.6581354472946848_2019-01-28_16:01_pcknot4.npy',
                        'val_result_F1_0.57371_L_0.6669414722463592_2019-01-27_00:46_pcknot5.npy',
                        'val_result_F1_0.57578_L_0.6985141360841068_2019-01-27_00:54_pcknot5.npy',
                        'val_result_F1_0.57623_L_0.6621931040825227_2019-01-28_00:32_pcknot5.npy',
                        'val_result_F1_0.57759_L_0.703442574330578_2019-01-28_00:15_pcbirger.npy',
                        'val_result_F1_0.58191_L_0.6831989158446978_2019-01-28_17:11_pcknot5.npy']

found_best_ensemble2 = ['val_result_F1_0.55141_L_0.7247291676338632_2019-01-28_10:12_pcknot2.npy',
                        'val_result_F1_0.55580_L_0.7056901221467318_2019-01-26_20:24_pcknot4.npy',
                        'val_result_F1_0.55930_L_0.6865916204641289_2019-01-27_16:14_pcbirger.npy',
                        'val_result_F1_0.55995_L_0.6895784042835879_2019-01-27_01:26_pcknot4.npy',
                        'val_result_F1_0.56315_L_0.6799039991718097_2019-01-28_16:35_pcbirger.npy',
                        'val_result_F1_0.56552_L_0.6853782425250486_2019-01-28_13:18_pcknot2.npy',
                        'val_result_F1_0.56750_L_0.6898565446440823_2019-01-26_20:31_pcknot4.npy',
                        'val_result_F1_0.56786_L_0.6935550871222328_2019-01-28_16:55_pcknot5.npy',
                        'val_result_F1_0.57423_L_0.7102468566180802_2019-01-28_17:03_pcknot5.npy',
                        'val_result_F1_0.57578_L_0.6985141360841068_2019-01-27_00:54_pcknot5.npy',
                        'val_result_F1_0.57759_L_0.703442574330578_2019-01-28_00:15_pcbirger.npy',
                        'val_result_F1_0.57948_L_0.6698856112670224_2019-01-28_08:24_pcknot5.npy']

#TRAIN_PRIORS = np.array([17.73049645, 67.45255894, 7.24554342, 7.57140119])
#DEV_PRIORS = np.array([6.86868687, 79.52861953, 5.52188552, 8.08080808])
#TEST_PRIORS = np.array([8.593322386425834, 80.78817733990148,  5.528188286808977,5.0903119868637114])/100

# TEST DATA
#TOP-N
#85.22167487684729 NO PRIOR !

#85.00273672687466 TRAIN PRIOR
#83. 68910782703887 TEST PRIOR

#EXC_K
#0.8549534756431308 NO PRIOR!
#0.8467432950191571 TRAIN PRIOR
#0.8347016967706623 TEST PRIOR

# VAL DATA
# 0.8296296296296296
# 0.8336700336700337 TRAIN PRIOR
# 0.8228956228956229 TEST PRIOR
def load_and_eval(lossfunction, produce_result=False, weights=None, prefix="val_", result_name="answer.json"
                  , strategy = "average_logits"):
    files = sorted(os.listdir("saved/ensemble/numpy_result/"))
    valid_ensemble_subset = [f"{prefix}{s[len('val_'):]}" for s in list(found_best_ensemble)]
    valid = [f for f in files if f.startswith(prefix) and f.endswith("npy")]

    result_files = [f for f in valid if "result" in f and f in valid_ensemble_subset]

    # assert len(result_files) == len(valid_ensemble_subset)
    def has_worst_substr(f):
        for w in worst_k:
            if w in f: return True
        return False

    #result_files = [f for f in valid if "result" in f and (not exclude_worst or not has_worst_substr(f))]
    print(result_files)
    print(f"Ensemble of {len(result_files)} files")
    label_file = [f for f in valid if "labels" in f][0]
    labels = np.load(os.path.join("saved/ensemble/numpy_result", label_file))
    result_matrices = [np.load(os.path.join("saved/ensemble/numpy_result", result_file)) for result_file in
                       result_files]
    results = np.array(result_matrices)

    tweet_ids = open(f"saved/ensemble/numpy_result/{prefix}ids.txt", "r").read().split()

    if strategy == "average_logits":
        feats = np.average(results, 0)
        results = torch.Tensor(feats).cuda()
    elif strategy == "sum_softmaxes":
        results = torch.Tensor(results).cuda()
        # Models x batch x classes
        results = F.softmax(results, -1)
        results = torch.sum(results, 0)
    elif strategy == "weighted_softmax_sum":
        results = torch.Tensor(results).cuda()
        # Models x batch x classes
        results = F.softmax(results, -1)
        for k in range(results.shape[0]): results[k] = results[k] * weights[k]
        results = torch.sum(results, 0)
    elif strategy == "avg_softmaxes":
        # Models x batch x classes
        results = torch.Tensor(results).cuda()
        results = F.softmax(results, -1)
        results = torch.mean(results, 0)

    else:
        return
    labels = torch.Tensor(labels).cuda().long()

    softmaxed_results = results if strategy == "avg_softmaxes" else F.softmax(results, -1)
    maxpreds, argmaxpreds = torch.max(softmaxed_results, dim=1)

    total_preds = list(argmaxpreds.cpu().numpy())
    if not produce_result:
        loss = lossfunction(results, labels)
        dev_loss = loss.item()
        total_labels = list(labels.cpu().numpy())

        correct_vec = argmaxpreds == labels
        total_correct = torch.sum(correct_vec).item()

        loss, acc = dev_loss, total_correct / results.shape[0]
        F1 = metrics.f1_score(total_labels, total_preds, average="macro")
        F1_cls = metrics.f1_score(total_labels, total_preds, average=None)
        return loss, acc, F1, tuple(F1_cls)
    else:
        answers = {"subtaskaenglish": dict(),
                   "subtaskbenglish": dict(),
                   "subtaskadanish": dict(),
                   "subtaskbdanish": dict(),
                   "subtaskarussian": dict(),
                   "subtaskbrussian": dict()
                   }
        task = "subtaskaenglish"

        for ix, p in enumerate(total_preds):
            answers[task][tweet_ids[ix]] = map_stance_label_to_s[p.item()]

        with open(result_name, "w") as  answer_file:
            json.dump(answers, answer_file)
        logging.info(f"Writing results into {result_name}")


def eval_F1__for_paper(lossfunction, prefix="val_"):
    files = sorted(os.listdir("saved/ensemble/numpy_result/"))
    valid = [f for f in files if f.startswith(prefix) and f.endswith("npy")]
    result_files = [f for f in valid if "result" in f]
    print(result_files)
    print(f"Ensemble of {len(result_files)} files")
    label_file = [f for f in valid if "labels" in f][0]
    labels = np.load(os.path.join("saved/ensemble/numpy_result", label_file))
    result_matrices = [torch.Tensor(np.load(os.path.join("saved/ensemble/numpy_result", result_file))).cuda() for
                       result_file in result_files]
    tweet_ids = open(f"saved/ensemble/numpy_result/{prefix}ids.txt", "r").read().split()
    labels = torch.Tensor(labels).cuda().long()
    F1s = []
    F1_c0 = []
    F1_c1 = []
    F1_c2 = []
    F1_c3 = []
    for f, model_results in zip(result_files, result_matrices):
        softmaxed_results = F.softmax(model_results, -1)
        maxpreds, argmaxpreds = torch.max(softmaxed_results, dim=1)

        total_preds = list(argmaxpreds.cpu().numpy())
        loss = lossfunction(model_results, labels)
        dev_loss = loss.item()
        total_labels = list(labels.cpu().numpy())

        correct_vec = argmaxpreds == labels
        total_correct = torch.sum(correct_vec).item()

        loss, acc = dev_loss, total_correct / softmaxed_results.shape[0]
        F1 = metrics.f1_score(total_labels, total_preds, average="macro")
        F1_cls = metrics.f1_score(total_labels, total_preds, average=None)
        F1s.append(F1)
        F1_c0.append(F1_cls[0])
        F1_c1.append(F1_cls[1])
        F1_c2.append(F1_cls[2])
        F1_c3.append(F1_cls[3])
        # print(f, loss, acc, F1)
    print(sum(F1s) / len(F1s))  # F1s
    print(statistics.stdev(F1s))
    print(sum(F1_c0) / len(F1_c0))
    print(sum(F1_c1) / len(F1_c1))
    print(sum(F1_c2) / len(F1_c2))
    print(sum(F1_c3) / len(F1_c3))


def find_best_ensemble_greedy():
    files = sorted(os.listdir("saved/ensemble/numpy"))
    valid = [f for f in files if f.startswith("val_") and f.endswith("npy")]
    result_files = sorted([f for f in valid if "result" in f], reverse=True)
    # print(result_files)
    label_file = [f for f in valid if "labels" in f][0]
    labels = np.load(os.path.join("saved/ensemble/numpy", label_file))
    result_matrices = {result_file: np.load(os.path.join("saved/ensemble/numpy", result_file)) for result_file in
                       result_files}

    allm = set(result_matrices.keys())
    available_matrices = allm.copy()
    ensemble_matrices = set()
    seed = random.choice(list(available_matrices))
    best_F1 = find_F1(labels, [result_matrices[seed]])
    while len(available_matrices) > 0:
        available_matrices = allm - ensemble_matrices
        # print(best_F1)
        found = False
        topickfrom = (list(available_matrices))
        random.shuffle(topickfrom)
        for m in topickfrom:
            candidate_matrices = ensemble_matrices.copy()
            candidate_matrices.add(m)
            F1 = find_F1(labels, [result_matrices[a] for a in candidate_matrices])

            if F1 > best_F1:
                best_F1 = F1
                ensemble_matrices.add(m)
                found = True
                break
        if not found:
            break

    # logging.info('\n'.join(ensemble_matrices))
    # logging.info(f"F1: {best_F1}")
    return best_F1, ensemble_matrices


def remove_worst_k(k=10):
    files = sorted(os.listdir("saved/ensemble/numpy"))
    valid = [f for f in files if f.startswith("val_") and f.endswith("npy")]
    result_files = sorted([f for f in valid if "result" in f], reverse=True)
    # print(result_files)
    label_file = [f for f in valid if "labels" in f][0]
    labels = np.load(os.path.join("saved/ensemble/numpy", label_file))
    result_matrices = {result_file: np.load(os.path.join("saved/ensemble/numpy", result_file)) for result_file in
                       result_files}

    ensemble_matrices = set(result_matrices.keys())
    best_F1 = find_F1(labels, [result_matrices[m] for m in ensemble_matrices])
    removed = 0
    to_remove = None
    while removed < k:
        for m in ensemble_matrices:
            F1 = find_F1(labels, [result_matrices[m] for m in ensemble_matrices - {m}])
            if F1 > best_F1:
                best_F1 = F1
                to_remove = m
        if to_remove is not None:
            ensemble_matrices.remove(to_remove)
            print(f"Removed :{to_remove}")
            # print(f"Best F1{best_F1}")
            to_remove = None
            removed += 1
        else:
            break

    # logging.info('\n'.join(ensemble_matrices))
    # logging.info(f"F1: {best_F1}")
    return best_F1, ensemble_matrices


def find_F1(labels, results):
    results = torch.Tensor(np.array(list(results))).cuda()
    # Models x batch x classes
    results = F.softmax(results, -1)
    results = torch.mean(results, 0)
    labels = torch.Tensor(labels).cuda().long()
    maxpreds, argmaxpreds = torch.max(results, dim=1)
    total_preds = list(argmaxpreds.cpu().numpy())
    total_labels = list(labels.cpu().numpy())
    F1 = metrics.f1_score(total_labels, total_preds, average="macro")
    return F1


if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath="logs/",
                  config_path="configurations/logging.yml")
    intent = "eval_F1_paper"
    if intent == "eval_F1_paper":
        eval_F1__for_paper(torch.nn.CrossEntropyLoss(
            weight=torch.Tensor([3.8043243885040283, 1.0, 9.309523582458496, 8.90886116027832]).cuda()),
            prefix="test_"
        )
    elif intent == "remove_k":
        # 5848667685861412
        F1, fs = remove_worst_k()
        logging.info('\n'.join(sorted(list(fs), reverse=True)))
        logging.info(F1)
        logging.info(f"LEN: {len(fs)}")
    elif intent == "eval":
        print(load_and_eval(torch.nn.CrossEntropyLoss(
            weight=torch.Tensor([3.8043243885040283, 1.0, 9.309523582458496, 8.90886116027832]).cuda()),
            weights=None, exclude_worst=True
        ))
    elif intent == "eval_gold_labels":
        print(load_and_eval(torch.nn.CrossEntropyLoss(
            weight=torch.Tensor([3.8043243885040283, 1.0, 9.309523582458496, 8.90886116027832]).cuda()),
            prefix="test_",
            weights=None
        ))
    elif intent == "run_on_test_data":
        load_and_eval(torch.nn.CrossEntropyLoss(
            weight=torch.Tensor([3.8043243885040283, 1.0, 9.309523582458496, 8.90886116027832]).cuda()),
            weights=None,
            prefix="test_",
            produce_result=True
        )
    elif intent == "find_subset":
        best_F1 = 0
        for k in range(10000):
            random.seed(k + random.randint(0, 1e10))
            F1, fs = find_best_ensemble_greedy()
            if F1 > best_F1:
                best_F1 = F1
                logging.info('\n'.join(sorted(list(fs), reverse=True)))
                logging.info(F1)
                logging.info(f"LEN: {len(fs)}")
# 0.8500273672687466
