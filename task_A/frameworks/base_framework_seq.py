import csv
import logging
import math
import os
import time
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F
import torchtext
from torch.nn.modules.loss import _Loss
from torchtext.data import BucketIterator, Iterator
from tqdm import tqdm

from modelutils import glorot_param_init
from task_A.datasets.RumourEvalDataset_Seq import RumourEval2019Dataset_Seq
from task_A.frameworks.base_framework import Base_Framework
from task_A.frameworks.bert_framework import map_stance_label_to_s
from utils import count_parameters, get_timestamp

__author__ = "Martin Fajčík"

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

    def run_epoch(self, model, lossfunction, optimizer, train_iter, config, verbose=False):
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

    def train(self, modelfunc):
        config = self.config

        fields = RumourEval2019Dataset_Seq.prepare_fields(text_field=lambda: torchtext.data.RawField())
        train_data, train_fields = self.build_dataset(config["train_data"], fields)
        dev_data, dev_fields = self.build_dataset(config["dev_data"], fields)

        torch.manual_seed(42)

        # No need to build vocab for baseline
        # but fo future work I wrote RumourEval2019Dataset that
        # requires vocab to be build

        device = torch.device("cuda:0" if config['cuda'] and
                                          torch.cuda.is_available() else "cpu")

        create_iter = lambda data: BucketIterator(data,
                                                  batch_size=config["hyperparameters"]["batch_size"], repeat=False,
                                                  device=device)
        train_iter = create_iter(train_data)
        dev_iter = create_iter(dev_data)

        logging.info(f"Train examples: {len(train_data.examples)}\nValidation examples: {len(dev_data.examples)}")

        model = modelfunc(self.config).to(device)

        glorot_param_init(model)
        logging.info(f"Model has {count_parameters(model)} trainable parameters.")
        logging.info(f"Manual seed {torch.initial_seed()}")
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=config["hyperparameters"]["learning_rate"],
                                     betas=[0.9, 0.999], eps=1e-8)

        # lossfunction = torch.nn.CrossEntropyLoss()
        # With L1
        def CE_wL1(preds, labels, lmb=0.01):
            def L1(model):
                accumulator = 0
                for p in filter(lambda p: p.requires_grad, model.parameters()):
                    accumulator += torch.sum(torch.abs(p))
                return accumulator

            return F.cross_entropy(preds, labels) + lmb * L1(model)

        lossfunction = CE_wL1
        start_time = time.time()
        try:
            best_val_loss = math.inf
            best_val_acc = 0
            for epoch in range(config["hyperparameters"]["epochs"]):
                train_loss, train_acc = self.run_epoch(model, lossfunction, optimizer, train_iter, config)
                validation_loss, validation_acc, val_acc_per_level = self.validate(model, lossfunction, dev_iter,
                                                                                   config)
                sorted_val_acc_pl = sorted(val_acc_per_level.items(), key=lambda x: int(x[0]))
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                if validation_acc > best_val_acc:
                    best_val_acc = validation_acc
                logging.info(
                    f"Epoch {epoch}, Validation loss|acc: {validation_loss:.6f}|{validation_acc:.6f} - (Best {best_val_loss:.4f}|{best_val_acc:4f})")
                logging.debug(
                    f"Epoch {epoch}, Validation loss|acc: {validation_loss:.6f}|{validation_acc:.6f} - (Best {best_val_loss:.4f}|{best_val_acc:4f})")
                logging.debug("\n".join([f"{k} - {v:.2f}" for k, v in sorted_val_acc_pl]))
                if validation_acc > self.save_treshold:
                    torch.save(model,
                               f"saved/checkpoint_{str(self.__class__)}_ACC_{validation_acc:.5f}_{get_timestamp()}.pt")
        except KeyboardInterrupt:
            logging.info('-' * 120)
            logging.info('Exit from training early.')
        finally:
            logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')

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

            if log_results:
                maxpreds, argmaxpreds = torch.max(F.softmax(pred_logits, -1), dim=1)
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
        if log_results:
            self.finalize_results_logging(csvf, loss, acc)
        if train_flag:
            model.train()
        return loss, acc, total_acc_per_level

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
