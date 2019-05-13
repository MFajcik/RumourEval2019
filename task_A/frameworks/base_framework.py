__author__ = "Martin Fajčík"

import json
import random
import logging
import math
import time
import torch
import torch.nn.functional as F

from sklearn import metrics
from torch.nn.modules.loss import _Loss
from torchtext.data import BucketIterator, Iterator
from tqdm import tqdm
from collections import defaultdict, Callable, Counter
from neural_bag.modelutils import glorot_param_init
from task_A.datasets.RumourEvalDataset_Branches import RumourEval2019Dataset_Branches
from utils.utils import count_parameters, get_timestamp


class Base_Framework:
    def __init__(self, config: dict, save_treshold=0.45):
        self.config = config
        self.save_treshold = save_treshold
        self.saveruns = False  # do not save if not explicitly stated

    def build_dataset(self, path, fields):
        return RumourEval2019Dataset_Branches(path, fields), {k: v for k, v in fields}

    def fit(self, modelfunc):
        config = self.config

        fields = RumourEval2019Dataset_Branches.prepare_fields(self.config["hyperparameters"]["sep_token"])
        train_data, train_fields = self.build_dataset(config["train_data"], fields)
        dev_data, dev_fields = self.build_dataset(config["dev_data"], fields)

        build_vocab = lambda field, *data: RumourEval2019Dataset_Branches.build_vocab(field,
                                                                                      self.config["hyperparameters"][
                                                                                          "sep_token"],
                                                                                      *data,
                                                                                      vectors=config["embeddings"],
                                                                                      vectors_cache=config[
                                                                                          "vector_cache"])
        build_vocab(train_fields['spacy_processed_text'], train_data, dev_data)
        self.vocab = train_fields['spacy_processed_text'].vocab
        device = torch.device("cuda:0" if config['cuda'] and
                                          torch.cuda.is_available() else "cpu")

        train_iter = BucketIterator(train_data,  # sort_key=lambda x: -len(x.text), sort=True,
                                    shuffle=True,
                                    batch_size=config["hyperparameters"]["batch_size"], repeat=False,
                                    device=device)
        dev_iter = BucketIterator(train_data, sort_key=lambda x: -len(x.text), sort=True,
                                  batch_size=config["hyperparameters"]["batch_size"], repeat=False,
                                  device=device)

        logging.info(f"Train examples: {len(train_data.examples)}\nValidation examples: {len(dev_data.examples)}")

        model = modelfunc(self.config, vocab=train_fields['spacy_processed_text'].vocab).to(device)

        glorot_param_init(model)
        logging.info(f"Model has {count_parameters(model)} trainable parameters.")
        logging.info(f"Manual seed {torch.initial_seed()}")
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=config["hyperparameters"]["learning_rate"],
                                     betas=[0.9, 0.999], eps=1e-8)

        # hardcoded weights precalculated for RumourEval 2019 train data
        lossfunction = torch.nn.CrossEntropyLoss(
            weight=torch.Tensor([3.8043243885040283, 1.0, 9.309523582458496, 8.90886116027832]).to(device))
        start_time = time.time()
        try:
            best_val_loss = math.inf
            best_val_acc = 0
            best_val_F1 = 0
            for epoch in range(config["hyperparameters"]["epochs"]):
                self.train(model, lossfunction, optimizer, train_iter, config)
                val_F1, validation_loss, validation_acc = self.validate(model, lossfunction, dev_iter, config)
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                if validation_acc > best_val_acc:
                    best_val_acc = validation_acc
                if val_F1 > best_val_F1:
                    best_val_F1 = val_F1
                logging.info(
                    f"Validation loss|acc|F1|BEST_F1: {validation_loss:.6f}|{validation_acc:.6f}|{val_F1:.6f}|{best_val_F1}")
                if validation_acc > self.save_treshold:
                    torch.save(model,
                               f"saved/checkpoint_{str(self.__class__)}_ACC_{validation_acc:.5f}_{get_timestamp()}.pt")
        except KeyboardInterrupt:
            logging.info('-' * 120)
            logging.info('Exit from training early.')
        finally:
            logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')

    def train(self, model, lossfunction, optimizer, train_iter, config, verbose=False):
        total_batches = len(train_iter.data()) // train_iter.batch_size
        if verbose:
            pbar = tqdm(total=total_batches)
        examples_so_far = 0
        train_loss = 0
        total_correct = 0

        already_seen = []
        valid_examples = 0
        for i, batch in enumerate(train_iter):
            pred_logits = model(batch)

            str_ids = [id for id_ in batch.string_id for id in id_]
            seen_mask = []
            for id in str_ids:
                if id in already_seen:
                    seen_mask.append(0)
                else:
                    seen_mask.append(1)
                    already_seen.append(id)

            seen_mask = torch.Tensor(seen_mask).byte()
            pred_logits = pred_logits.view((-1, 4))
            labels = batch.stance_labels.view(-1)
            mask = labels > -1
            masked_preds, masked_labels = pred_logits[mask, :], labels[mask]
            masked_preds, masked_labels = masked_preds[seen_mask, :], masked_labels[seen_mask]
            valid_examples += masked_preds.shape[0]
            loss = lossfunction(masked_preds, masked_labels)

            optimizer.zero_grad()
            loss.backward()
            if model.encoder is not None:
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), config["hyperparameters"]["RNN_clip"])
            optimizer.step()

            total_correct += self.calculate_correct(masked_preds, masked_labels)

            examples_so_far += 1
            train_loss += loss.item()
            if verbose:
                pbar.set_description(
                    f"Train loss: {train_loss / (i + 1):.4f}, Train acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)
        print(f"Total unique examples {valid_examples}")
        return train_loss / train_iter.batch_size, total_correct / examples_so_far

    def fit_multiple(self, modelfunc: Callable, trials: int = 20):
        """
        Runs training multiple times and prints the average statistics
        :param modelfunc: model constructor
        :param trials: number of times to run the model
        """

        results = []
        for i in range(trials):
            torch.manual_seed(random.randint(1, 1e8))
            results.append(self.fit(modelfunc))
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

    @torch.no_grad()
    def validate(self, model: torch.nn.Module, lossfunction: _Loss, dev_iter: Iterator, config: dict, verbose=False):
        train_flag = model.training
        model.eval()

        total_batches = len(dev_iter.data()) // dev_iter.batch_size
        if verbose:
            pbar = tqdm(total=total_batches)
        examples_so_far = 0
        dev_loss = 0
        total_correct = 0
        total_labels = []
        total_preds = []
        already_seen = []
        for i, batch in enumerate(dev_iter):
            pred_logits = model(batch)
            str_ids = [id for id_ in batch.string_id for id in id_]

            seen_mask = []
            for id in str_ids:
                if id in already_seen:
                    seen_mask.append(0)
                else:
                    seen_mask.append(1)
                    already_seen.append(id)

            # mask used as index need to be byte tensor
            # otherwise, the indices 0 and 1 will be picked instead!
            seen_mask = torch.Tensor(seen_mask).byte()

            pred_logits = pred_logits.view((-1, 4))
            labels = batch.stance_labels.view(-1)
            mask = labels > -1
            masked_preds, masked_labels = pred_logits[mask, :], labels[mask]
            masked_preds, masked_labels = masked_preds[seen_mask, :], masked_labels[seen_mask]

            loss = lossfunction(masked_preds, masked_labels)

            total_correct += self.calculate_correct(masked_preds, masked_labels)

            maxpreds, argmaxpreds = torch.max(F.softmax(masked_preds, -1), dim=1)
            total_preds += list(argmaxpreds.cpu().numpy())
            total_labels += list(masked_labels.cpu().numpy())

            examples_so_far += len(masked_labels)
            dev_loss += loss.item()
            if verbose:
                pbar.set_description(
                    f"dev loss: {dev_loss / (i + 1):.4f}, dev acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)
        if train_flag:
            model.train()
        F1 = metrics.f1_score(total_labels, total_preds, average="macro")
        return F1, dev_loss / dev_iter.batch_size, total_correct / examples_so_far

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

    def finalize_results_logging(self, csvf, workbook):
        raise NotImplementedError

    def init_result_logging(self):
        raise NotImplementedError
