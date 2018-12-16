import logging
import math
import time

import torch
from pytorch_pretrained_bert import BertAdam, BertTokenizer
from torch.nn.modules.loss import _Loss
from torchtext.data import BucketIterator, Iterator
from tqdm import tqdm

from RumourEvalDataset_Triplets import RumourEval2019Dataset_BERTTriplets
from task_A.frameworks.base_framework import Base_Framework
from utils import count_parameters, get_timestamp


class BERT_Framework(Base_Framework):

    def __init__(self, config: dict):
        super().__init__(config)
        self.save_treshold = 0.78
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="./.BERTcache")

    def run_epoch(self, model, lossfunction, optimizer, train_iter, config, verbose=False):
        total_batches = len(train_iter.data()) // train_iter.batch_size
        if verbose:
            pbar = tqdm(total=total_batches)
        examples_so_far = 0
        train_loss = 0
        total_correct = 0
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            pred_logits = model(batch)

            loss = lossfunction(pred_logits, batch.stance_label)

            loss.backward()
            optimizer.step()

            total_correct += self.calculate_correct(pred_logits, batch.stance_label)
            examples_so_far += len(batch.stance_label)
            train_loss += loss.item()
            if verbose:
                pbar.set_description(
                    f"Train loss:"
                    f" {train_loss / (i + 1):.4f}, Train acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)
        return train_loss / total_batches, total_correct / examples_so_far

    def train(self, modelfunc):
        config = self.config

        fields = RumourEval2019Dataset_BERTTriplets.prepare_fields()
        train_data = RumourEval2019Dataset_BERTTriplets(config["train_data"], fields, self.tokenizer,
                                                        max_length=config["hyperparameters"]["max_length"])
        dev_data = RumourEval2019Dataset_BERTTriplets(config["dev_data"], fields, self.tokenizer,
                                                      max_length=config["hyperparameters"]["max_length"])

        torch.manual_seed(1570055016034928672 & ((1 << 63) - 1))

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
            for epoch in range(config["hyperparameters"]["epochs"]):
                train_loss, train_acc = self.run_epoch(model, lossfunction, optimizer, train_iter, config)
                validation_loss, validation_acc = self.validate(model, lossfunction, dev_iter, config)
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                if validation_acc > best_val_acc:
                    best_val_acc = validation_acc
                logging.info(
                    f"Epoch {epoch}, Validation loss|acc: {validation_loss:.6f}|{validation_acc:.6f} - (Best {best_val_loss:.4f}|{best_val_acc:4f})")
                if validation_acc > self.save_treshold:
                    torch.save(model,
                               f"saved/checkpoint_{str(self.__class__)}_ACC_{validation_acc:.5f}_{get_timestamp()}.pt")
        except KeyboardInterrupt:
            logging.info('-' * 120)
            logging.info('Exit from training early.')
        finally:
            logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')

    def validate(self, model: torch.nn.Module, lossfunction: _Loss, dev_iter: Iterator, config: dict, verbose=False):
        train_flag = model.training
        model.eval()

        total_batches = len(dev_iter.data()) // dev_iter.batch_size
        if verbose:
            pbar = tqdm(total=total_batches)
        examples_so_far = 0
        dev_loss = 0
        total_correct = 0
        for i, batch in enumerate(dev_iter):
            pred_logits = model(batch)

            loss = lossfunction(pred_logits, batch.stance_label)

            total_correct += self.calculate_correct(pred_logits, batch.stance_label)
            examples_so_far += len(batch.stance_label)
            dev_loss += loss.item()
            if verbose:
                pbar.set_description(
                    f"dev loss: {dev_loss / (i + 1):.4f}, dev acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)
        if train_flag:
            model.train()
        return dev_loss / total_batches, total_correct / examples_so_far
