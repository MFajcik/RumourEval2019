import csv
import json
import logging
import math
import os
import random
import time
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F
import torchtext
from sklearn import metrics
from torch.nn.modules.loss import _Loss
from torchtext.data import BucketIterator, Iterator
from tqdm import tqdm

from neural_bag.modelutils import glorot_param_init
from task_A.datasets.RumourEvalDataset_Seq import RumourEval2019Dataset_Seq
from task_A.frameworks.base_framework import Base_Framework
from utils.utils import setup_logging, map_stance_label_to_s
from task_A.frameworks.self_att_with_bert_tokenizing import SelfAtt_BertTokenizing_Framework
from utils.utils import count_parameters, get_timestamp

__author__ = "Martin Fajčík"
import socket

step = 0


# FIXME: learn special embedding tokens
# RNN baseline (Best 2.1256|0.854494) drop 0.6, FC 2, fc_size 300
# No RNN base (Best 2.1018|0.857347)
# textonly  (Best 3.6824|0.690799)
# textonly
# textonly + embopt (Best 3.9955|0.676177)

# textonly 0.714337 after text preprocessing
class Base_Framework_SEQ(Base_Framework):
    def build_dataset(self, path, fields):
        return RumourEval2019Dataset_Seq(path, fields), {k: v for k, v in fields}

    def train(self, model, lossfunction, optimizer, train_iter, config, verbose=False):
        total_batches = len(train_iter.data()) // train_iter.batch_size
        if verbose:
            pbar = tqdm(total=total_batches)
        examples_so_far = 0
        train_loss = 0
        total_correct = 0

        optimizer.zero_grad()
        for i, batch in enumerate(train_iter):
            updated = False
            pred_logits = model(batch)

            loss = lossfunction(pred_logits, batch.stance_label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            total_correct += self.calculate_correct(pred_logits, batch.stance_label)
            examples_so_far += len(batch.stance_label)
            train_loss += loss.item()
            if verbose:
                pbar.set_description(
                    f"train loss:"
                    f" {train_loss / (i + 1):.4f}, train acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)
        loss, acc = train_loss / total_batches, total_correct / examples_so_far
        return loss, acc

    def run_training(self, modelfunc):
        config = self.config

        fields = RumourEval2019Dataset_Seq.prepare_fields(text_field=lambda: torchtext.data.RawField())
        train_data, train_fields = self.build_dataset(config["train_data"], fields)
        dev_data, dev_fields = self.build_dataset(config["dev_data"], fields)
        test_data, test_fields = self.build_dataset(config["test_data"], fields)

        # No need to build vocab for baseline
        # but fo future work I wrote RumourEval2019Dataset that
        # requires vocab to be build

        device = torch.device("cuda:0" if config['cuda'] and
                                          torch.cuda.is_available() else "cpu")

        create_iter = lambda data: BucketIterator(data, sort_key=lambda x: -len(x.spacy_processed_text), sort=True,
                                                  batch_size=config["hyperparameters"]["batch_size"], repeat=False,
                                                  device=device)
        train_iter = create_iter(train_data)
        dev_iter = create_iter(dev_data)
        test_iter = create_iter(test_data)

        logging.info(f"Train examples: {len(train_data.examples)}\nValidation examples: {len(dev_data.examples)}")

        model = modelfunc(self.config).to(device)

        glorot_param_init(model)
        logging.info(f"Model has {count_parameters(model)} trainable parameters.")
        logging.info(f"Manual seed {torch.initial_seed()}")
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=config["hyperparameters"]["learning_rate"],
                                     betas=[0.9, 0.999], eps=1e-8)

        # weights = SelfAtt_BertTokenizing_Framework.get_class_weights(train_data.examples, "stance_label", 4,
        #                                                              min_fraction=1)
        # logging.info("class weights")
        # logging.info(f"{str(weights.numpy().tolist())}")
        # lossfunction = torch.nn.CrossEntropyLoss(weight=weights.to(device))
        lossfunction = torch.nn.CrossEntropyLoss()
        # # With L1
        # def CE_wL1(preds, labels, lmb=0.01):
        #     def L1(model):
        #         accumulator = 0
        #         for p in filter(lambda p: p.requires_grad, model.parameters()):
        #             accumulator += torch.sum(torch.abs(p))
        #         return accumulator
        #
        #     return F.cross_entropy(preds, labels) + lmb * L1(model)
        #
        # lossfunction = CE_wL1
        start_time = time.time()
        best_val_loss = math.inf
        best_val_acc = 0
        best_val_F1 = 0
        best_F1_loss, best_loss_F1 = 0, 0
        bestF1_testF1 = 0
        bestF1_test_F1s = [0, 0, 0, 0]
        best_val_F1s = [0, 0, 0, 0]
        start_time = time.time()
        try:

            # self.predict("answer_BERT_textnsource.json", model, dev_iter)
            best_val_los_epoch = -1
            early_stop_after = 8  # steps
            for epoch in range(config["hyperparameters"]["epochs"]):
                self.epoch = epoch
                self.train(model, lossfunction, optimizer, train_iter, config)
                train_loss, train_acc, _, train_F1, train_allF1s = self.validate(model, lossfunction, train_iter,
                                                                                 config,
                                                                                 log_results=False)
                validation_loss, validation_acc, val_acc_per_level, val_F1, val_allF1s = self.validate(model,
                                                                                                       lossfunction,
                                                                                                       dev_iter,
                                                                                                       config,
                                                                                                       log_results=False)
                saved = False
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    best_val_los_epoch = epoch
                    best_loss_F1 = val_F1
                    if val_F1 > self.save_treshold:
                        # Map to CPU before saving, because this requires additional memory /for some reason/
                        model.to(torch.device("cpu"))
                        torch.save(model,
                                   f"saved/BIG_checkpoint_{str(self.__class__)}_F1"
                                   f"_{val_F1:.5f}_L_{validation_loss}_{get_timestamp()}_{socket.gethostname()}.pt")
                        model.to(device)
                        saved = True

                if validation_acc > best_val_acc:
                    best_val_acc = validation_acc
                    best_F1_loss = validation_loss
                    best_val_F1s = val_allF1s
                    test_loss, test_acc, test_acc_per_level, bestF1_testF1, bestF1_test_F1s = self.validate(model,
                                                                                                            lossfunction,
                                                                                                            test_iter,
                                                                                                            config,
                                                                                                            log_results=False)

                    if val_F1 > self.save_treshold and not saved:
                        # Map to CPU before saving, because this requires additional memory /for some reason/
                        model.to(torch.device("cpu"))
                        torch.save(model,
                                   f"saved/BIG_checkpoint_{str(self.__class__)}_F1"
                                   f"_{val_F1:.5f}_L_{validation_loss}_{get_timestamp()}_{socket.gethostname()}.pt")
                        model.to(device)

                if val_F1 > best_val_F1:
                    best_val_F1 = val_F1


                logging.info(
                    f"Epoch {epoch}, Training loss|acc|F1: {train_loss:.6f}|{train_acc:.6f}|{train_F1:.6f}")
                logging.info(
                    f"Epoch {epoch}, Validation loss|acc|F1: {validation_loss:.6f}|{validation_acc:.6f}|{val_F1:.6f} - "
                    f"(Best {best_val_loss:.4f}|{best_val_acc:4f}|{best_val_F1}|{bestF1_testF1})")

                if validation_loss > best_val_loss and epoch > best_val_los_epoch + early_stop_after:
                    logging.info("Early stopping...")
                    break
        except KeyboardInterrupt:
            logging.info('-' * 120)
            logging.info('Exit from training early.')
        finally:
            logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')
        return {
            "best_loss": best_val_loss,
            "best_acc": best_val_acc,
            "best_F1": best_val_F1,
            "bestF1_loss": best_F1_loss,
            "bestloss_F1": best_loss_F1,
            "bestACC_testACC": test_acc,
            "bestF1_testF1": bestF1_testF1,
            "val_bestF1_C1F1": best_val_F1s[0],
            "val_bestF1_C2F1": best_val_F1s[1],
            "val_bestF1_C3F1": best_val_F1s[2],
            "val_bestF1_C4F1": best_val_F1s[3],
            "test_bestF1_C1F1": bestF1_test_F1s[0],
            "test_bestF1_C2F1": bestF1_test_F1s[1],
            "test_bestF1_C3F1": bestF1_test_F1s[2],
            "test_bestF1_C4F1": bestF1_test_F1s[3]
        }

    def fit(self, modelfunc, trials=20):
        results = []
        for i in range(trials):
            torch.manual_seed(random.randint(1, 1e8))
            results.append(self.run_training(modelfunc))
        logging.info("Results:")
        for i in range(trials):
            logging.info(f"{i} :{json.dumps(results[i])}")

        logging.info("*" * 20 + "AVG" + "*" * 20)
        avg = Counter(results[0])
        for i in range(1, trials): avg += Counter(results[i])
        for key in avg:
            avg[key] /= trials
        logging.info(json.dumps(avg))
        logging.info("*" * 20 + "AVG ends" + "*" * 20)

    def validate(self, model: torch.nn.Module, lossfunction: _Loss, dev_iter: Iterator, config: dict, verbose=False,
                 log_results=False):
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
            maxpreds, argmaxpreds = torch.max(F.softmax(pred_logits, -1), dim=1)
            total_preds += list(argmaxpreds.cpu().numpy())
            total_labels += list(batch.stance_label.cpu().numpy())
            if verbose:
                pbar.set_description(
                    f"dev loss: {dev_loss / (i + 1):.4f}, dev acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)

            if log_results:
                pred_s = list(argmaxpreds.cpu().numpy())
                target_s = list(batch.stance_label.cpu().numpy())
                correct_s = list((argmaxpreds == batch.stance_label).cpu().numpy())
                prob_s = [f"{x:.2f}" for x in
                          list(maxpreds.cpu().detach().numpy())]

                for i in range(len(correct_s)):
                    writer.writerow([correct_s[i],
                                     batch.id[i],
                                     batch.tweet_id[i],
                                     branch_levels[i],
                                     map_stance_label_to_s[target_s[i]],
                                     map_stance_label_to_s[pred_s[i]],
                                     prob_s[i],
                                     batch.raw_text[i]])

        loss, acc = dev_loss / total_batches, total_correct / examples_so_far
        total_acc_per_level = {depth: total_correct_per_level.get(depth, 0) / total for depth, total in
                               total_per_level.items()}

        F1 = metrics.f1_score(total_labels, total_preds, average="macro")
        allF1s = metrics.f1_score(total_labels, total_preds, average=None).tolist()

        if log_results:
            self.finalize_results_logging(csvf, loss, acc)
        if train_flag:
            model.train()
        return loss, acc, total_acc_per_level, F1, allF1s

    def finalize_results_logging(self, csvf, loss, acc):
        csvf.close()
        os.rename(self.TMP_FNAME,
                  f"introspection/introspection_baseline_{str(self.__class__)}_A{acc:.6f}_L{loss:.6f}.tsv", )

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
        self.TMP_FNAME = f"introspection/introspection_{str(self.__class__)}.tsv"
        csvf = open(self.TMP_FNAME, mode="w")
        writer = csv.writer(csvf, delimiter='\t')
        writer.writerow(self.__class__.RESULT_HEADER)
        return csvf, writer

    def calculate_correct(self, pred_logits: torch.Tensor, labels: torch.Tensor, levels=None):
        preds = torch.argmax(pred_logits, dim=1)
        correct_vec = preds == labels
        if not levels:
            return torch.sum(correct_vec).item()
        else:
            sums_per_level = defaultdict(lambda: 0)
            for level, correct in zip(levels, correct_vec):
                sums_per_level[level] += correct.item()
            return torch.sum(correct_vec).item(), sums_per_level
