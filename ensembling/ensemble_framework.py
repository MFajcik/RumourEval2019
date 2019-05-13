import csv
import logging
import os
import socket
import time
from collections import Counter, defaultdict, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertAdam
from sklearn import metrics
from torch.nn.modules.loss import _Loss
from torchtext.data import BucketIterator
from tqdm import tqdm

from ensembling.ensemble_helper import load_and_eval
from ensembling.secondary_cls import SecondaryCls
from neural_bag.modelutils import glorot_param_init
from task_A.datasets.RumourEvalDataset_BERT import RumourEval2019Dataset_BERTTriplets
from task_A.frameworks.base_framework import Base_Framework
from task_A.frameworks.self_att_with_bert_tokenizing import SelfAtt_BertTokenizing_Framework

# this is a list of best ensemble predictions
# found via find_best_ensemble_greedy method from ensemble_helper.py
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


class Ensemble_Framework(Base_Framework):
    def __init__(self, config: dict):
        super().__init__(config)

        # Uncomment this to run logistic regression L2 classifier
        # self.create_l2_optim(config, torch.nn.CrossEntropyLoss(
        #    weight=torch.Tensor([3.8043243885040283, 1.0, 9.309523582458496, 8.90886116027832]).cuda()))
        # sys.exit()

        self.save_treshold = 999
        self.modeltype = config["modeltype"]
        self.tokenizer = BertTokenizer.from_pretrained(self.modeltype, cache_dir="./.BERTcache",
                                                       do_lower_case=True)

    # In these experiments I have tried to learn an optimal combination of ensemble predictions via L2 classifier
    # I have tried various approaches (combining predictions / pre-softmax scores) but reached bad results
    # As for a space constraints, this experiment did not make it into the paper

    def create_l2_optim(self, config, lossfunction):
        files = sorted(os.listdir("saved/ensemble/numpy"))
        train_prefix = "val_"
        valid = [f for f in files if f.startswith(train_prefix) and f.endswith("npy")]
        valid_ensemble_subset = [f"{train_prefix}{s[len('val_'):]}" for s in list(found_best_ensemble)]
        #        result_files = [f for f in valid if "result" in f]
        result_files = [f for f in valid if "result" in f and f in valid_ensemble_subset]
        assert len(result_files) == len(valid_ensemble_subset)
        logging.debug(result_files)
        label_file = [f for f in valid if "labels" in f][0]
        labels = np.load(os.path.join("saved/ensemble/numpy", label_file))
        result_matrices = [np.load(os.path.join("saved/ensemble/numpy", result_file)) for result_file in result_files]
        results = np.array(result_matrices)

        # experiment 2, try softmaxing logits first
        results = torch.Tensor(results)
        results = F.softmax(results, -1).numpy()
        results = torch.Tensor(np.concatenate(results, -1)).cuda()

        # experiment 1, traing LR on logits
        # results = np.concatenate(results, -1)
        # results = torch.Tensor(results).cuda()

        labels = torch.Tensor(labels).cuda().long()

        valid = [f for f in files if f.startswith("val_") and f.endswith("npy")]

        valid_ensemble_subset = [f"val_{s[len('val_'):]}" for s in list(found_best_ensemble)]
        #        result_files = [f for f in valid if "result" in f]
        result_files = [f for f in valid if "result" in f and f in valid_ensemble_subset]
        assert len(result_files) == len(valid_ensemble_subset)
        logging.debug(result_files)
        label_file = [f for f in valid if "labels" in f][0]
        dev_labels = np.load(os.path.join("saved/ensemble/numpy", label_file))
        dev_results = np.array(
            [np.load(os.path.join("saved/ensemble/numpy", result_file)) for result_file in result_files])

        dev_results = torch.Tensor(dev_results)
        dev_results = F.softmax(dev_results, -1).numpy()
        dev_results = torch.Tensor(np.concatenate(dev_results, -1)).cuda()
        # dev_results = np.concatenate(dev_results, -1)
        # dev_results = torch.Tensor(dev_results).cuda()

        dev_labels = torch.Tensor(dev_labels).cuda().long()
        total_labels = list(dev_labels.cpu().numpy())

        ens_best_F1 = 0
        ens_best_distribution = None
        l = torch.nn.CrossEntropyLoss(
            weight=torch.Tensor([3.8043243885040283, 1.0, 9.309523582458496, 8.90886116027832]).cuda())
        for _ in range(1000):

            F1, distribution = self.run_LR_training(config, dev_labels, dev_results, labels, lossfunction,
                                                    results, total_labels)

            logging.info(f"New Best F1: {F1}")
            if F1 > ens_best_F1:
                _, _, e_f1 = load_and_eval(l,
                                           weights=distribution
                                           )
                if e_f1 != F1:
                    F1, distribution = self.run_LR_training(config, dev_labels, dev_results, labels, lossfunction,
                                                            results, total_labels)
                ens_best_F1 = F1
                ens_best_distribution = distribution
                logging.debug(f"New Best F1: {ens_best_F1}")
                logging.debug(ens_best_distribution)

    def run_LR_training(self, config, dev_labels, dev_results, labels, lossfunction, results, total_labels):
        model = SecondaryCls(config).cuda()
        glorot_param_init(model)
        optimizer = BertAdam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=config["hyperparameters"]["learning_rate"], weight_decay=0.02)
        best_distribution = None
        best_F1 = 0
        for i in range(1000):
            pred_logits = model(results)
            loss = lossfunction(pred_logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            dev_pred_logits = model(dev_results)
            dev_loss = lossfunction(dev_pred_logits, dev_labels)
            maxpreds, argmaxpreds = torch.max(F.softmax(dev_pred_logits, -1), dim=1)
            total_preds = list(argmaxpreds.cpu().numpy())
            correct_vec = argmaxpreds == dev_labels
            total_correct = torch.sum(correct_vec).item()
            loss, acc = dev_loss, total_correct / results.shape[0]
            F1 = metrics.f1_score(total_labels, total_preds, average="macro")
            if F1 > best_F1:
                best_F1 = F1
                best_distribution = F.softmax(model.a)

            # logging.info(
            #     f"Validation loss|acc|F1|BEST: {loss:.6f}|{acc:.6f}|{F1:.6f} || {best_F1} || ")
        return best_F1, best_distribution

    def fit(self, modelfunc):
        config = self.config

        fields = RumourEval2019Dataset_BERTTriplets.prepare_fields_for_text()
        train_data = RumourEval2019Dataset_BERTTriplets(config["train_data"], fields, self.tokenizer,
                                                        max_length=config["hyperparameters"]["max_length"])
        dev_data = RumourEval2019Dataset_BERTTriplets(config["dev_data"], fields, self.tokenizer,
                                                      max_length=config["hyperparameters"]["max_length"])
        test_data = RumourEval2019Dataset_BERTTriplets(config["test_data"], fields, self.tokenizer,
                                                       max_length=config["hyperparameters"]["max_length"])

        # torch.manual_seed(5246727901370826861 & ((1 << 63) - 1))
        # torch.manual_seed(40)

        # device = torch.device("cpu")
        device = torch.device("cuda:0" if config['cuda'] and
                                          torch.cuda.is_available() else "cpu")

        create_iter = lambda data: BucketIterator(data, sort_key=lambda x: -len(x.text), sort=True,
                                                  batch_size=config["hyperparameters"]["batch_size"],
                                                  repeat=False,
                                                  device=device)

        dev_iter = create_iter(dev_data)
        test_iter = create_iter(test_data)

        logging.info(f"Train examples: {len(train_data.examples)}\nValidation examples: {len(dev_data.examples)}")
        logging.info(f"Test examples: {len(test_data.examples)}")

        checkpoints = os.listdir("saved/ensemble/")
        modelpaths = sorted([f"saved/ensemble/{ch}" for ch in checkpoints if ch.endswith(".pt")])
        logging.info(f"Running ensemble of {len(modelpaths)} models")

        models = []

        weights = SelfAtt_BertTokenizing_Framework.get_class_weights(train_data.examples, "stance_label", 4,
                                                                     min_fraction=1)
        logging.info("class weights")
        logging.info(f"{str(weights.numpy().tolist())}")
        lossfunction = torch.nn.CrossEntropyLoss(weight=weights.to(device))  # .to(device))

        soft_ensemble = False
        build_predictions = True
        check_f1s = False
        eval_from_npy = False
        train_logreg = False
        if train_logreg:
            start_time = time.time()
            try:
                model = self.create_l2_optim(config, lossfunction)
            except KeyboardInterrupt:
                logging.info('-' * 120)
                logging.info('Exit from training early.')

            finally:
                logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')
        elif eval_from_npy:
            start_time = time.time()
            try:
                load_and_eval(lossfunction)
            except KeyboardInterrupt:
                logging.info('-' * 120)
                logging.info('Exit from training early.')
            finally:
                logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')
        elif build_predictions:
            # generate .npy predictions from models
            start_time = time.time()
            try:

                for idx, model_path in enumerate(modelpaths):
                    pretrained_model = torch.load(
                        model_path)
                    model = modelfunc.from_pretrained(self.modeltype, cache_dir="./.BERTcache",
                                                      state_dict=pretrained_model.state_dict()
                                                      ).to(device)
                    model.dropout = pretrained_model.dropout
                    logging.info("MODEL: " + model_path)
                    suffix = model_path[model_path.index("F1"):model_path.index(".pt")]

                    # generate prediction scores from train data
                    # train_loss, train_acc, _, train_F1 = self.build_results(idx, model, suffix,
                    #                                                         lossfunction,
                    #                                                         train_iter,
                    #                                                         config,
                    #                                                         prefix="train_", )

                    # generate prediction scores from val data
                    # validation_loss, validation_acc, val_acc_per_level, val_F1 = self.build_results(idx, model, suffix,
                    #                                                                                 lossfunction,
                    #                                                                                 dev_iter,
                    #                                                                                 config,
                    #                                                                                 prefix="val_")

                    # generate prediction scores from test data
                    self.build_results(idx, model, suffix,
                                       lossfunction,
                                       test_iter,
                                       config,
                                       prefix="test_",
                                       do_not_evaluate=True)
                    # logging.info(
                    #     f"Training loss|acc|F1: {train_loss:.6f}|{train_acc:.6f}|{train_F1:.6f}")
                    # logging.info(
                    #     f"Validation loss|acc|F1: {validation_loss:.6f}|{validation_acc:.6f}|{val_F1:.6f}")
            except KeyboardInterrupt:
                logging.info('-' * 120)
                logging.info('Exit from training early.')
            finally:
                logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')
        elif soft_ensemble:
            start_time = time.time()
            try:
                for modelpath in modelpaths:
                    pretrained_model = torch.load(
                        modelpath)
                    model = modelfunc.from_pretrained(self.modeltype, cache_dir="./.BERTcache",
                                                      state_dict=pretrained_model.state_dict()
                                                      ).to(device)
                    model.dropout = pretrained_model.dropout
                    models.append(model)

                validation_loss, validation_acc, val_acc_per_level, val_F1 = self.validate_models(models,
                                                                                                  lossfunction,
                                                                                                  dev_iter,
                                                                                                  config,
                                                                                                  log_results=False)
                logging.info(
                    f"Validation loss|acc|F1: {validation_loss:.6f}|{validation_acc:.6f}|{val_F1:.6f}")
            except KeyboardInterrupt:
                logging.info('-' * 120)
                logging.info('Exit from training early.')
            finally:
                logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')

        elif check_f1s:
            # pretrained_model = None
            start_time = time.time()
            try:
                for idx, modelpath in enumerate(modelpaths):
                    pretrained_model = torch.load(
                        modelpath)
                    model = modelfunc.from_pretrained(self.modeltype, cache_dir="./.BERTcache",
                                                      state_dict=pretrained_model.state_dict()
                                                      ).to(device)
                    model.dropout = pretrained_model.dropout
                    logging.info(f"Model: {checkpoints[idx]}")
                    # self.predict(f"answer_BERTF1_textonly_{idx}.json", model, dev_iter)
                    # train_loss, train_acc, _, train_F1 = self.validate(model, lossfunction, train_iter, config,
                    #                                                    log_results=False)
                    validation_loss, validation_acc, val_acc_per_level, val_F1 = self.validate(model, lossfunction,
                                                                                               dev_iter,
                                                                                               config,
                                                                                               log_results=False)

                    # logging.info(
                    #     f"Training loss|acc|F1: {train_loss:.6f}|{train_acc:.6f}|{train_F1:.6f}")
                    logging.info(
                        f"Validation loss|acc|F1: {validation_loss:.6f}|{validation_acc:.6f}|{val_F1:.6f}")

            except KeyboardInterrupt:
                logging.info('-' * 120)
                logging.info('Exit from training early.')
            finally:
                logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')

    def build_results(self, k, model: torch.nn.Module, suffix, lossfunction: _Loss, dev_iter: Iterator, config: dict,
                      prefix="val_", verbose=False, do_not_evaluate=False):
        if not os.path.exists("saved/ensemble/numpy/"):
            os.makedirs("saved/ensemble/numpy/")
        train_flag = model.training
        model.eval()

        total_examples = len(dev_iter.data())

        results = np.zeros((total_examples, 4))

        total_batches = len(dev_iter.data()) // dev_iter.batch_size
        if verbose:
            pbar = tqdm(total=total_batches)
        examples_so_far = 0
        dev_loss = 0
        total_correct = 0
        total_correct_per_level = Counter()
        total_per_level = defaultdict(lambda: 0)
        total_labels = []
        total_preds = []
        ids = []
        for idx, batch in enumerate(dev_iter):
            pred_logits = model(batch)

            numpy_logits = pred_logits.cpu().detach().numpy()  # bsz x classes
            step_size = numpy_logits.shape[0]
            write_index = idx * dev_iter.batch_size
            results[write_index: write_index + step_size] = numpy_logits
            ids += batch.tweet_id

            if not do_not_evaluate:
                loss = lossfunction(pred_logits, batch.stance_label)
                branch_levels = [id.split(".", 1)[-1] for id in batch.branch_id]
                for branch_depth in branch_levels: total_per_level[branch_depth] += 1
                correct, correct_per_level = self.calculate_correct(pred_logits, batch.stance_label,
                                                                    levels=branch_levels)
                total_correct += correct
                total_correct_per_level += correct_per_level

                examples_so_far += len(batch.stance_label)
                dev_loss += loss.item()

                maxpreds, argmaxpreds = torch.max(F.softmax(pred_logits, -1), dim=1)
                total_preds += list(argmaxpreds.cpu().numpy())
                total_labels += list(batch.stance_label.cpu().numpy())

            if verbose:
                pbar.set_description(
                    f"dev loss: {dev_loss / (idx + 1):.4f}, dev acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)

        if not do_not_evaluate:
            loss, acc = dev_loss / total_batches, total_correct / examples_so_far
            total_acc_per_level = {depth: total_correct_per_level.get(depth, 0) / total for depth, total in
                                   total_per_level.items()}
            F1 = metrics.f1_score(total_labels, total_preds, average="macro")
        np.save(f"saved/ensemble/numpy/{prefix}result_{suffix}.npy", results)
        if k == 0:
            np.save(f"saved/ensemble/numpy/{prefix}labels.npy", np.array(total_labels))
            with open(f"saved/ensemble/numpy/{prefix}ids.txt", "w") as f:
                f.write('\n'.join(ids))
        if train_flag:
            model.train()
        if do_not_evaluate:
            return
        return loss, acc, total_acc_per_level, F1

    def validate_models(self, models, lossfunction: _Loss, dev_iter: Iterator, config: dict, verbose=True,
                        log_results=True):
        train_flags = [model.training for model in models]

        for model in models: model.eval()

        total_batches = len(dev_iter.data()) // dev_iter.batch_size
        if verbose:
            pbar = tqdm(total=total_batches)
        if log_results:
            csvf, writer = self.init_result_logging()
        examples_so_far = 0
        dev_loss = 0
        total_correct = 0
        total_correct_per_level = Counter()
        total_per_level = defaultdict(lambda: 0)
        total_labels = []
        total_preds = []
        for i, batch in enumerate(dev_iter):
            pred_logits = None
            for model in models:
                # model = model.cuda()
                pred_logits_per_model = model(batch)
                if pred_logits is None:
                    pred_logits = F.softmax(pred_logits_per_model, -1)
                else:
                    pred_logits += F.softmax(pred_logits_per_model, -1)

                # model = model.cpu()
                # torch.cuda.empty_cache()

            loss = lossfunction(pred_logits, batch.stance_label)

            branch_levels = [id.split(".", 1)[-1] for id in batch.branch_id]
            for branch_depth in branch_levels: total_per_level[branch_depth] += 1
            correct, correct_per_level = self.calculate_correct(pred_logits, batch.stance_label, levels=branch_levels)
            total_correct += correct
            total_correct_per_level += correct_per_level

            examples_so_far += len(batch.stance_label)
            dev_loss += loss.item()
            if verbose:
                pbar.set_description(
                    f"dev loss: {dev_loss / (i + 1):.4f}, dev acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)

            maxpreds, argmaxpreds = torch.max(F.softmax(pred_logits, -1), dim=1)
            total_preds += list(argmaxpreds.cpu().numpy())
            total_labels += list(batch.stance_label.cpu().numpy())
            if log_results:
                text_s = [' '.join(self.tokenizer.convert_ids_to_tokens(batch.text[i].cpu().numpy())) for i in
                          range(batch.text.shape[0])]
                pred_s = list(argmaxpreds.cpu().numpy())
                target_s = list(batch.stance_label.cpu().numpy())
                correct_s = list((argmaxpreds == batch.stance_label).cpu().numpy())
                prob_s = [f"{x:.2f}" for x in
                          list(maxpreds.cpu().detach().numpy())]

                assert len(text_s) == len(pred_s) == len(correct_s) == len(
                    target_s) == len(prob_s)
                for i in range(len(text_s)):
                    writer.writerow([correct_s[i],
                                     batch.id[i],
                                     batch.tweet_id[i],
                                     branch_levels[i],
                                     map_stance_label_to_s[target_s[i]],
                                     map_stance_label_to_s[pred_s[i]],
                                     prob_s[i],
                                     batch.raw_text[i],
                                     text_s[i]])

        loss, acc = dev_loss / total_batches, total_correct / examples_so_far
        total_acc_per_level = {depth: total_correct_per_level.get(depth, 0) / total for depth, total in
                               total_per_level.items()}
        F1 = metrics.f1_score(total_labels, total_preds, average="macro")
        if log_results:
            self.finalize_results_logging(csvf, loss, F1)

        for train_flag in train_flags:
            if train_flag:
                model.fit()
        return loss, acc, total_acc_per_level, F1

    def validate(self, model: torch.nn.Module, lossfunction: _Loss, dev_iter: Iterator, config: dict, verbose=False,
                 log_results=True):
        train_flag = model.training
        model.eval()

        total_batches = len(dev_iter.data()) // dev_iter.batch_size
        if verbose:
            pbar = tqdm(total=total_batches)
        if log_results:
            csvf, writer = self.init_result_logging()
        examples_so_far = 0
        dev_loss = 0
        total_correct = 0
        total_correct_per_level = Counter()
        total_per_level = defaultdict(lambda: 0)
        total_labels = []
        total_preds = []
        for i, batch in enumerate(dev_iter):
            pred_logits = model(batch)

            loss = lossfunction(pred_logits, batch.stance_label)

            branch_levels = [id.split(".", 1)[-1] for id in batch.branch_id]
            for branch_depth in branch_levels: total_per_level[branch_depth] += 1
            correct, correct_per_level = self.calculate_correct(pred_logits, batch.stance_label, levels=branch_levels)
            total_correct += correct
            total_correct_per_level += correct_per_level

            examples_so_far += len(batch.stance_label)
            dev_loss += loss.item()
            if verbose:
                pbar.set_description(
                    f"dev loss: {dev_loss / (i + 1):.4f}, dev acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)

            maxpreds, argmaxpreds = torch.max(F.softmax(pred_logits, -1), dim=1)
            total_preds += list(argmaxpreds.cpu().numpy())
            total_labels += list(batch.stance_label.cpu().numpy())
            if log_results:
                text_s = [' '.join(self.tokenizer.convert_ids_to_tokens(batch.text[i].cpu().numpy())) for i in
                          range(batch.text.shape[0])]
                pred_s = list(argmaxpreds.cpu().numpy())
                target_s = list(batch.stance_label.cpu().numpy())
                correct_s = list((argmaxpreds == batch.stance_label).cpu().numpy())
                prob_s = [f"{x:.2f}" for x in
                          list(maxpreds.cpu().detach().numpy())]

                assert len(text_s) == len(pred_s) == len(correct_s) == len(
                    target_s) == len(prob_s)
                for i in range(len(text_s)):
                    writer.writerow([correct_s[i],
                                     batch.id[i],
                                     batch.tweet_id[i],
                                     branch_levels[i],
                                     map_stance_label_to_s[target_s[i]],
                                     map_stance_label_to_s[pred_s[i]],
                                     prob_s[i],
                                     batch.raw_text[i],
                                     text_s[i]])

        loss, acc = dev_loss / total_batches, total_correct / examples_so_far
        total_acc_per_level = {depth: total_correct_per_level.get(depth, 0) / total for depth, total in
                               total_per_level.items()}
        F1 = metrics.f1_score(total_labels, total_preds, average="macro")
        if log_results:
            self.finalize_results_logging(csvf, loss, F1)
        if train_flag:
            model.train()
        return loss, acc, total_acc_per_level, F1

    def finalize_results_logging(self, csvf, loss, f1):
        csvf.close()
        os.rename(self.TMP_FNAME, f"introspection/introspection"
        f"_{str(self.__class__)}_A{f1:.6f}_L{loss:.6f}_{socket.gethostname()}.tsv", )

    RESULT_HEADER = ["Correct",
                     "data_id",
                     "tweet_id",
                     "branch_level",
                     "Ground truth",
                     "Prediction",
                     "Confidence",
                     "Text",
                     "Processed_Text"]

    def init_result_logging(self):
        self.TMP_FNAME = f"introspection/TMP_introspection_{str(self.__class__)}_{socket.gethostname()}.tsv"
        csvf = open(self.TMP_FNAME, mode="w")
        writer = csv.writer(csvf, delimiter='\t')
        writer.writerow(self.__class__.RESULT_HEADER)
        return csvf, writer
