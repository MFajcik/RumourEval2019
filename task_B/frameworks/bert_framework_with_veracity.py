import csv
import json
import logging
import math
import os
import socket
import time
import torch
import torch.nn.functional as F

from collections import Counter, defaultdict
from pytorch_pretrained_bert import BertAdam, BertTokenizer
from torch.nn.modules.loss import _Loss
from torchtext.data import BucketIterator, Iterator
from tqdm import tqdm
from task_A.datasets.RumourEvalDataset_BERT import RumourEval2019Dataset_BERTTriplets_with_Tags
from task_A.frameworks.base_framework import Base_Framework
from utils.utils import map_stance_label_to_s
from utils.utils import count_parameters, get_timestamp

__author__ = "Martin Fajčík"

"""
This file contains code for experiments with predicting veracity when taking only source experiments

Result: As expected, model overfits heavily and we did not used these results
"""


class BERT_Framework(Base_Framework):
    def __init__(self, config: dict):
        super().__init__(config)
        self.save_treshold = 0.83
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="./.BERTcache",
                                                       do_lower_case=True)

    def train(self, model, lossfunction, optimizer, train_iter, config, verbose=False):
        total_batches = len(train_iter.data()) // train_iter.batch_size
        if verbose:
            pbar = tqdm(total=total_batches)
        examples_so_far = 0
        train_loss = 0
        total_correct = 0

        update_ratio = config["hyperparameters"]["true_batch_size"] // config["hyperparameters"]["batch_size"]
        optimizer.zero_grad()

        # This flag controls whether it is time to update gradients
        # when accumulating the gradients
        updated = False

        for i, batch in enumerate(train_iter):
            updated = False
            pred_logits, pred_logits_veracity = model(batch)
            mask = batch.veracity_label > 0
            has_source_samples = pred_logits_veracity[mask].shape[0] > 0
            loss = (lossfunction(pred_logits, batch.stance_label) +
                    (lossfunction(pred_logits_veracity[mask],
                                  batch.veracity_label[mask]) if has_source_samples else 0)) \
                   / update_ratio
            loss.backward()

            if (i + 1) % update_ratio == 0:
                optimizer.step()
                optimizer.zero_grad()
                updated = True

            total_correct += self.calculate_correct(pred_logits, batch.stance_label)
            examples_so_far += len(batch.stance_label)
            train_loss += loss.item()
            if verbose:
                pbar.set_description(
                    f"train loss:"
                    f" {train_loss / (i + 1):.4f}, train acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)

        if not updated:
            optimizer.step()
            optimizer.zero_grad()

        return train_loss / total_batches, total_correct / examples_so_far

    def predict(self, fname: str, model: torch.nn.Module, dev_iter: Iterator, task="subtaskaenglish"):
        train_flag = model.training
        model.eval()
        answers = {"subtaskaenglish": dict(),
                   "subtaskbenglish": dict(),
                   "subtaskadanish": dict(),
                   "subtaskbdanish": dict(),
                   "subtaskarussian": dict(),
                   "subtaskbrussian": dict()
                   }
        for i, batch in enumerate(dev_iter):
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

    def fit(self, modelfunc):
        config = self.config

        fields = RumourEval2019Dataset_BERTTriplets_with_Tags.prepare_fields_for_text()
        train_data = RumourEval2019Dataset_BERTTriplets_with_Tags(config["train_data"], fields, self.tokenizer,
                                                                  max_length=config["hyperparameters"]["max_length"])
        dev_data = RumourEval2019Dataset_BERTTriplets_with_Tags(config["dev_data"], fields, self.tokenizer,
                                                                max_length=config["hyperparameters"]["max_length"])

        device = torch.device("cuda:0" if config['cuda'] and
                                          torch.cuda.is_available() else "cpu")

        create_iter = lambda data: BucketIterator(data, sort_key=lambda x: -len(x.text), sort=True,
                                                  # shuffle=True,
                                                  batch_size=config["hyperparameters"]["batch_size"], repeat=False,
                                                  device=device)
        train_iter = create_iter(train_data)
        dev_iter = create_iter(dev_data)

        logging.info(f"Train examples: {len(train_data.examples)}\nValidation examples: {len(dev_data.examples)}")

        model = modelfunc.from_pretrained("bert-base-uncased", cache_dir="./.BERTcache").to(device)
        logging.info(f"Model has {count_parameters(model)} trainable parameters.")
        logging.info(f"Manual seed {torch.initial_seed()}")
        optimizer = BertAdam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=config["hyperparameters"]["learning_rate"])

        lossfunction = torch.nn.CrossEntropyLoss()
        start_time = time.time()
        try:
            best_val_loss = math.inf
            best_val_acc = 0
            best_val_ver_acc = 0

            for epoch in range(config["hyperparameters"]["epochs"]):
                self.epoch = epoch

                train_loss, train_acc = self.train(model, lossfunction, optimizer, train_iter, config)
                validation_loss, validation_acc, val_acc_per_level, veracity_acc = self.validate(model, lossfunction,
                                                                                                 dev_iter,
                                                                                                 config,
                                                                                                 log_results=False)
                sorted_val_acc_pl = sorted(val_acc_per_level.items(), key=lambda x: int(x[0]))

                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                if validation_acc > best_val_acc:
                    best_val_acc = validation_acc
                if veracity_acc > best_val_ver_acc:
                    best_val_ver_acc = veracity_acc

                logging.info(
                    f"Epoch {epoch}, Validation loss|acc: {validation_loss:.6f}|{validation_acc:.6f} - (Best {best_val_loss:.4f}|{best_val_acc:4f})")
                logging.info(
                    f"Epoch {epoch}, Veracity acc:{veracity_acc:.6f} - (Best {best_val_ver_acc:.4f})")

                logging.debug(
                    f"Epoch {epoch}, Validation loss|acc: {validation_loss:.6f}|{validation_acc:.6f} - (Best {best_val_loss:.4f}|{best_val_acc:4f})")
                logging.debug("\n".join([f"{k} - {v:.2f}" for k, v in sorted_val_acc_pl]))
                if validation_acc > self.save_treshold:
                    model.to(torch.device("cpu"))
                    torch.save(model,
                               f"saved/checkpoint_{str(self.__class__)}_ACC_{validation_acc:.5f}_{get_timestamp()}.pt")
                    model.to(device)
        except KeyboardInterrupt:
            logging.info('-' * 120)
            logging.info('Exit from training early.')
        finally:
            logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')

    def validate(self, model: torch.nn.Module, lossfunction: _Loss, dev_iter: Iterator, config: dict, verbose=False,
                 log_results=True):
        train_flag = model.training
        model.eval()

        total_batches = len(dev_iter.data()) // dev_iter.batch_size
        if verbose:
            pbar = tqdm(total=total_batches)
        if log_results:
            csvf, writer = self.init_result_logging()
        dev_loss = 0
        examples_so_far = 0
        total_correct = 0
        examples_so_far_v = 0
        total_correct_v = 0
        total_correct_per_level = Counter()
        total_per_level = defaultdict(lambda: 0)
        for i, batch in enumerate(dev_iter):
            pred_logits, pred_logits_veracity = model(batch)

            mask = batch.veracity_label > 0
            has_source_samples = pred_logits_veracity[mask].shape[0] > 0

            loss = (lossfunction(pred_logits, batch.stance_label) +
                    (lossfunction(pred_logits_veracity[mask],
                                  batch.veracity_label[mask]) if has_source_samples else 0))

            branch_levels = [id.split(".", 1)[-1] for id in batch.branch_id]
            for branch_depth in branch_levels: total_per_level[branch_depth] += 1

            correct, correct_per_level = self.calculate_correct(pred_logits, batch.stance_label, levels=branch_levels)
            total_correct += correct
            total_correct_per_level += correct_per_level
            examples_so_far += len(batch.stance_label)
            dev_loss += loss.item()
            if has_source_samples:
                correct_v = self.calculate_correct(pred_logits_veracity[mask], batch.veracity_label[mask])
                total_correct_v += correct_v
                examples_so_far_v += len(batch.veracity_label[mask])

            if verbose:
                pbar.set_description(
                    f"dev loss: {dev_loss / (i + 1):.4f}, dev acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)

            if log_results:
                maxpreds, argmaxpreds = torch.max(F.softmax(pred_logits, -1), dim=1)
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

        veracity_acc = total_correct_v / examples_so_far_v
        if log_results:
            self.finalize_results_logging(csvf, loss, acc)
        if train_flag:
            model.train()
        return loss, acc, total_acc_per_level, veracity_acc

    def finalize_results_logging(self, csvf, loss, acc):
        csvf.close()
        os.rename(self.TMP_FNAME, f"introspection/introspection"
        f"_{str(self.__class__)}_A{acc:.6f}_L{loss:.6f}_{socket.gethostname()}.tsv", )

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
        self.TMP_FNAME = f"introspection/introspection_{str(self.__class__)}_{socket.gethostname()}.tsv"
        csvf = open(self.TMP_FNAME, mode="w")
        writer = csv.writer(csvf, delimiter='\t')
        writer.writerow(self.__class__.RESULT_HEADER)
        return csvf, writer
