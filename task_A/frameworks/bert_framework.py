__author__ = "Martin Fajčík"

import csv
import json
import logging
import math
import os
import socket
import time
import torch
import torch.nn.functional as F
import _csv
import _io

from pytorch_pretrained_bert import BertAdam, BertTokenizer
from sklearn import metrics
from torch.nn.modules.loss import _Loss
from torchtext.data import BucketIterator, Iterator
from torchtext.data.batch import Batch
from tqdm import tqdm
from task_A.datasets.RumourEvalDataset_BERT import RumourEval2019Dataset_BERTTriplets
from task_A.frameworks.base_framework import Base_Framework
from utils.utils import count_parameters, get_timestamp, map_stance_label_to_s, get_class_weights
from collections import Counter, defaultdict
from typing import Callable, Tuple, Dict, List


class BERT_Framework(Base_Framework):
    """
    Framework implementing BERT training with input pattern:
    [CLS]src post. prev post[SEP]target post[SEP]
    This is our best model, submitted to RumourEval2019 competition, referred to as BERT_{big} in the paper
    """

    def __init__(self, config: dict, save_treshold: float = 0.52):
        super().__init__(config, save_treshold)
        self.init_tokenizer()

    def init_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.config["variant"], cache_dir="./.BERTcache",
                                                       do_lower_case=True)

    def fit(self, modelfunc: Callable, skip_logging_nepochs: int = 5) -> dict:
        """
        Trains the model and executes early stopping
        :param modelfunc: model constructor
        :param skip_logging_nepochs: skip writing predictions into csv for initial skip_logging_nepochs epochs
        :return statistics of trained model
        """

        config = self.config

        fields = RumourEval2019Dataset_BERTTriplets.prepare_fields_for_text()
        train_data = RumourEval2019Dataset_BERTTriplets(config["train_data"], fields, self.tokenizer,
                                                        max_length=config["hyperparameters"]["max_length"])
        dev_data = RumourEval2019Dataset_BERTTriplets(config["dev_data"], fields, self.tokenizer,
                                                      max_length=config["hyperparameters"]["max_length"])
        test_data = RumourEval2019Dataset_BERTTriplets(config["test_data"], fields, self.tokenizer,
                                                       max_length=config["hyperparameters"]["max_length"])

        device = torch.device("cuda:0" if config['cuda'] and
                                          torch.cuda.is_available() else "cpu")

        train_iter = BucketIterator(train_data, sort_key=lambda x: -len(x.text), sort=True,
                                    shuffle=False,
                                    batch_size=config["hyperparameters"]["batch_size"], repeat=False,
                                    device=device)
        create_noshuffle_iter = lambda data: BucketIterator(data, sort_key=lambda x: -len(x.text), sort=True,
                                                            shuffle=False,
                                                            batch_size=config["hyperparameters"]["batch_size"],
                                                            repeat=False,
                                                            device=device)
        dev_iter = create_noshuffle_iter(dev_data)
        test_iter = create_noshuffle_iter(test_data)

        model = modelfunc.from_pretrained("bert-base-uncased", cache_dir="./.BERTcache").to(device)

        logging.info(f"Train examples: {len(train_data.examples)}\nValidation examples: {len(dev_data.examples)}")
        logging.info(f"Model has {count_parameters(model)} trainable parameters.")
        logging.info(f"Manual seed {torch.initial_seed()}")

        optimizer = BertAdam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=config["hyperparameters"]["learning_rate"])

        # Calculate weights for current data distribution
        weights = get_class_weights(train_data.examples, "stance_label", 4)

        logging.info("class weights")
        logging.info(f"{str(weights.numpy().tolist())}")
        lossfunction = torch.nn.CrossEntropyLoss(weight=weights.to(device))
        start_time = time.time()

        # Init counters and flags
        best_val_loss = math.inf
        best_val_acc = 0
        best_val_F1 = 0
        best_F1_loss, best_loss_F1 = 0, 0
        bestF1_testF1 = 0
        bestF1_testacc = 0
        bestF1_test_F1s = [0, 0, 0, 0]
        best_val_F1s = [0, 0, 0, 0]
        start_time = time.time()
        best_val_loss_epoch = -1
        try:
            # Uncomment to run prediction
            # self.predict("answer_BERTF1_textonly.json", model, dev_iter)
            for epoch in range(config["hyperparameters"]["epochs"]):
                self.epoch = epoch
                # this loss is computed during training, during active dropouts etc so it wont be similar to validation loss
                # but computing it second time over all training data is slow
                # You can call validate on train_iter if you wish to have proper training loss

                train_loss, train_acc = self.train(model, lossfunction, optimizer, train_iter, config)
                validation_loss, validation_acc, val_acc_per_level, val_F1, val_allF1s = self.validate(model,
                                                                                                       lossfunction,
                                                                                                       dev_iter,
                                                                                                       config,
                                                                                                       log_results=False)
                # accuracies per level
                sorted_val_acc_pl = sorted(val_acc_per_level.items(), key=lambda x: int(x[0]))

                saved = False
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    best_val_loss_epoch = epoch
                    if val_F1 > self.save_treshold and self.saveruns:
                        # Map to CPU before saving, because this requires additional memory /for some reason/
                        model.to(torch.device("cpu"))
                        torch.save(model,
                                   f"saved/BIG_checkpoint_{str(self.__class__)}_F1"
                                   f"_{val_F1:.5f}_L_{validation_loss}_{get_timestamp()}_{socket.gethostname()}.pt")
                        model.to(device)
                        saved = True

                if validation_acc > best_val_acc:
                    best_val_acc = validation_acc

                if val_F1 > best_val_F1:
                    best_val_F1 = val_F1
                    test_loss, bestF1_testacc, test_acc_per_level, bestF1_testF1, bestF1_test_F1s = self.validate(model,
                                                                                                                  lossfunction,
                                                                                                                  test_iter,
                                                                                                                  config,
                                                                                                                  log_results=False)

                    if val_F1 > self.save_treshold and not saved and self.saveruns:
                        # Map to CPU before saving, because this requires additional memory /for some reason/
                        model.to(torch.device("cpu"))
                        torch.save(model,
                                   f"saved/BIG_checkpoint_{str(self.__class__)}_F1"
                                   f"_{val_F1:.5f}_L_{validation_loss}_{get_timestamp()}_{socket.gethostname()}.pt")
                        model.to(device)
                # info logging
                logging.info(
                    f"Epoch {epoch}, Training loss|acc: {train_loss:.6f}|{train_acc:.6f}")
                logging.info(
                    f"Epoch {epoch}, Validation loss|acc|F1: {validation_loss:.6f}|{validation_acc:.6f}|{val_F1:.6f} - "
                    f"(Best {best_val_loss:.4f}|{best_val_acc:4f}|{best_val_F1})\n Best Test F1 - {bestF1_testF1}")

                # debug logging
                logging.debug(
                    f"Epoch {epoch}, Validation loss|acc|F1: {validation_loss:.6f}|{validation_acc:.6f}|{val_F1:.6f} - "
                    f"(Best {best_val_loss:.4f}|{best_val_acc:4f}|{best_val_F1})")
                logging.debug("\n".join([f"{k} - {v:.2f}" for k, v in sorted_val_acc_pl]))

                if validation_loss > best_val_loss and epoch > best_val_loss_epoch + self.config["early_stop_after"]:
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
            "bestACC_testACC": bestF1_testacc,
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

    def train(self, model: torch.nn.Module, lossfunction: _Loss, optimizer: torch.optim.Optimizer,
              train_iter: Iterator, config: dict, verbose=False) -> Tuple[float, float]:
        """
        :param model: model inherited from torch.nn.Module
        :param lossfunction:
        :param optimizer:
        :param train_iter:
        :param config:
        :param verbose: whether to print verbose outputs at stdout
        :return: train loss and train accuracy
        """
        if verbose:
            pbar = tqdm(total=len(train_iter.data()) // train_iter.batch_size)

        # Initialize accumulators & flags
        examples_so_far = 0
        train_loss = 0
        total_correct = 0
        N = 0
        updated = False

        # I case of gradient accumulalation, how often should gradient be updated
        update_ratio = config["hyperparameters"]["true_batch_size"] // config["hyperparameters"]["batch_size"]

        optimizer.zero_grad()
        for i, batch in enumerate(train_iter):
            updated = False
            pred_logits = model(batch)
            loss = lossfunction(pred_logits, batch.stance_label) / update_ratio
            loss.backward()

            if (i + 1) % update_ratio == 0:
                optimizer.step()
                optimizer.zero_grad()
                updated = True

            # Update accumulators
            train_loss += loss.item()
            N += 1 if not hasattr(lossfunction, "weight") \
                else sum([lossfunction.weight[k].item() for k in batch.stance_label])
            total_correct += self.calculate_correct(pred_logits, batch.stance_label)
            examples_so_far += len(batch.stance_label)

            if verbose:
                pbar.set_description(
                    f"train loss:"
                    f" {train_loss / (i + 1):.4f}, train acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)

        # Do the last step if needed with what has been accumulated
        if not updated:
            optimizer.step()
            optimizer.zero_grad()

        return train_loss / N, total_correct / examples_so_far

    @torch.no_grad()
    def validate(self, model: torch.nn.Module, lossfunction: _Loss, dev_iter: Iterator, config: dict, verbose=False,
                 log_results=True) -> Tuple[float, float, Dict[str, float], float, List[float]]:
        """

        :param model: model inherited from torch.nn.Module
        :param lossfunction:
        :param dev_iter:
        :param config:
        :param verbose:
        :param log_results: whether to print verbose outputs at stdout
        :return: validation loss, validation accuracy, validation accuracies per level, validation F1, per class F1s
        """

        train_flag = model.training
        model.eval()

        if verbose:
            pbar = tqdm(total=len(dev_iter.data()) // dev_iter.batch_size)
        if log_results:
            csvf, writer = self.init_result_logging()

        # initialize accumulators & flags
        examples_so_far = 0
        dev_loss = 0
        total_correct = 0
        N = 0
        total_correct_per_level = Counter()
        total_per_level = defaultdict(lambda: 0)
        total_labels = []
        total_preds = []

        for i, batch in enumerate(dev_iter):
            pred_logits = model(batch)
            loss = lossfunction(pred_logits, batch.stance_label)
            maxpreds, argmaxpreds = torch.max(F.softmax(pred_logits, -1), dim=1)

            # compute branch statistics
            branch_levels = [id.split(".", 1)[-1] for id in batch.branch_id]
            for branch_depth in branch_levels: total_per_level[branch_depth] += 1

            # compute correct and correct per branch depth
            correct, correct_per_level = self.calculate_correct(pred_logits, batch.stance_label, levels=branch_levels)
            total_correct += correct
            total_correct_per_level += correct_per_level
            examples_so_far += len(batch.stance_label)
            dev_loss += loss.item()
            N += 1 if not hasattr(lossfunction, "weight") \
                else sum([lossfunction.weight[k].item() for k in batch.stance_label])
            total_preds += list(argmaxpreds.cpu().numpy())
            total_labels += list(batch.stance_label.cpu().numpy())

            if verbose:
                pbar.set_description(
                    f"dev loss: {dev_loss / (i + 1):.4f}, dev acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)

            if log_results:
                self.log_to_csv(argmaxpreds, batch, branch_levels, maxpreds, writer)

        loss, acc = dev_loss / N, total_correct / examples_so_far
        total_acc_per_level = {depth: total_correct_per_level.get(depth, 0) / total for depth, total in
                               total_per_level.items()}
        F1 = metrics.f1_score(total_labels, total_preds, average="macro").item()
        allF1s = metrics.f1_score(total_labels, total_preds, average=None).tolist()
        if log_results:
            self.finalize_results_logging(csvf, loss, F1)
        if train_flag:
            model.train()
        return loss, acc, total_acc_per_level, F1, allF1s

    def log_to_csv(self, argmaxpreds: torch.Tensor, batch: Batch, branch_levels: list, maxpreds: torch.Tensor,
                   writer: _csv.writer):
        # convert to strings
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

    @torch.no_grad()
    def predict(self, fname: str, model: torch.nn.Module, data_iter: Iterator, task="subtaskaenglish"):
        """
        Predict data passed via Iterator into file, following SemEval Task's format
        :param fname: File name fo write predictions into
        :param model: Model used for predictions
        :param data_iter: Data iterator
        :param task: What task data belong to
        """

        train_flag = model.training
        model.eval()
        answers = {"subtaskaenglish": dict(),
                   "subtaskbenglish": dict(),
                   "subtaskadanish": dict(),
                   "subtaskbdanish": dict(),
                   "subtaskarussian": dict(),
                   "subtaskbrussian": dict()
                   }
        for i, batch in enumerate(data_iter):
            pred_logits = model(batch)
            preds = torch.argmax(pred_logits, dim=1)
            preds = list(preds.cpu().numpy())

            for ix, p in enumerate(preds):
                answers[task][batch.tweet_id[ix]] = map_stance_label_to_s[p.item()]

        with open(fname, "w") as  answer_file:
            json.dump(answers, answer_file)
        if train_flag:
            model.train()
        logging.info(f"Writing results into {fname}")

    # Columns of resulting csv file
    RESULT_HEADER = ["Correct",
                     "data_id",
                     "tweet_id",
                     "branch_level",
                     "Ground truth",
                     "Prediction",
                     "Confidence",
                     "Text",
                     "Processed_Text"]

    def finalize_results_logging(self, csvf: _io.TextIOWrapper, loss: float, f1: float):
        """
        Finalize writing of result CSV
        :param csvf: CSV file handle
        :param loss: loss reached with result
        :param f1: F1 reached with result
        """

        csvf.close()
        os.rename(self.TMP_FNAME, f"introspection/introspection"
        f"_{str(self.__class__)}_A{f1:.6f}_L{loss:.6f}_{socket.gethostname()}.tsv", )

    def init_result_logging(self) -> Tuple[_io.TextIOWrapper, _csv.writer]:
        """
        Initialize writing of result CSV
        :return: csv file handle and csv writer
        """
        # final name should contain metadata about loss/F1, which are not available during initialization
        # use temporary name, which will be changed upon finalization
        self.TMP_FNAME = f"introspection/TMP_introspection_{str(self.__class__)}_{socket.gethostname()}.tsv"
        csvf = open(self.TMP_FNAME, mode="w")
        writer = csv.writer(csvf, delimiter='\t')
        writer.writerow(self.__class__.RESULT_HEADER)
        return csvf, writer
