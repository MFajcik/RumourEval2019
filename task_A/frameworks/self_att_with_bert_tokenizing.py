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
import xlsxwriter

from collections import Counter, defaultdict
from pytorch_pretrained_bert import BertAdam, BertTokenizer
from sklearn import metrics
from torch.nn.modules.loss import _Loss
from torchtext.data import BucketIterator, Iterator
from tqdm import tqdm
from task_A.datasets.RumourEvalDataset_BERT import RumourEval2019Dataset_BERTTriplets
from task_A.frameworks.base_framework import Base_Framework
from task_A.frameworks.text_framework_branch import Text_Framework
from task_A.frameworks.text_framework_seq import Text_Framework_Seq
from utils.utils import count_parameters, get_timestamp, map_stance_label_to_s, get_class_weights


class SelfAtt_BertTokenizing_Framework(Base_Framework):
    """
    This framework runs model based on BiLSTM+SelfAttention using embeddings obtained via BERT pretraining.
    In our paper, we refer to this model as to "BiLSTM+SelfAtt"
    """

    def __init__(self, config: dict, save_treshold: int = 0.46):
        super().__init__(config, save_treshold)
        self.init_tokenizer()

    def init_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.config["variant"], cache_dir="./.BERTcache",
                                                       do_lower_case=True)

    def train(self, model, lossfunction, optimizer, train_iter, config, verbose=False):
        if verbose:
            pbar = tqdm(total=len(train_iter.data()) // train_iter.batch_size)
        examples_so_far = 0
        train_loss = 0
        total_correct = 0
        N = 0

        optimizer.zero_grad()
        for i, batch in enumerate(train_iter):
            pred_logits, attention = model(batch)
            loss = lossfunction(pred_logits, batch.stance_label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            total_correct += self.calculate_correct(pred_logits, batch.stance_label)
            examples_so_far += len(batch.stance_label)
            train_loss += loss.item()
            N += 1 if not hasattr(lossfunction, "weight") \
                else sum([lossfunction.weight[k].item() for k in batch.stance_label])
            if verbose:
                pbar.set_description(
                    f"train loss:"
                    f" {train_loss / (i + 1):.4f}, train acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)

        return train_loss / N, total_correct / examples_so_far

    @torch.no_grad()
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

        logging.info(f"Train examples: {len(train_data.examples)}\nValidation examples: {len(dev_data.examples)}")

        model = modelfunc.from_pretrained(self.config["variant"], cache_dir="./.BERTcache")
        model.extra_init(config, self.tokenizer)
        model = model.to(device)
        logging.info(f"Model has {count_parameters(model) - count_parameters(model.bert)} trainable parameters.")
        logging.info(f"Manual seed {torch.initial_seed()}")
        optimizer = BertAdam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=config["hyperparameters"]["learning_rate"])

        weights = get_class_weights(train_data.examples, "stance_label", 4)
        lossfunction = torch.nn.CrossEntropyLoss(weight=weights.to(device))

        start_time = time.time()
        best_val_loss = math.inf
        best_val_acc = 0
        best_val_F1 = 0
        best_F1_loss, best_loss_F1 = 0, 0
        bestF1_testF1 = 0
        bestF1_testacc = 0
        bestF1_test_F1s = [0, 0, 0, 0]
        best_val_F1s = [0, 0, 0, 0]
        start_time = time.time()
        try:

            # self.predict("answer_BERT_textnsource.json", model, dev_iter)
            best_val_loss_epoch = -1
            early_stop_after = 4  # steps
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
                    best_val_loss_epoch = epoch
                    best_loss_F1 = val_F1
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
                    best_F1_loss = validation_loss
                    best_val_F1s = val_allF1s
                    test_loss, bestF1_testacc, test_acc_per_level, bestF1_testF1, bestF1_test_F1s = self.validate(model,
                                                                                                                  lossfunction,
                                                                                                                  test_iter,
                                                                                                                  config,
                                                                                                                  log_results=val_F1 > 0.46)
                    if val_F1 > self.save_treshold and not saved and self.saveruns:
                        # Map to CPU before saving, because this requires additional memory /for some reason/
                        model.to(torch.device("cpu"))
                        torch.save(model,
                                   f"saved/BIG_checkpoint_{str(self.__class__)}_F1"
                                   f"_{val_F1:.5f}_L_{validation_loss}_{get_timestamp()}_{socket.gethostname()}.pt")
                        model.to(device)

                logging.info(
                    f"Epoch {epoch}, Training loss|acc|F1: {train_loss:.6f}|{train_acc:.6f}|{train_F1:.6f}")
                logging.info(
                    f"Epoch {epoch}, Validation loss|acc|F1: {validation_loss:.6f}|{validation_acc:.6f}|{val_F1:.6f} - "
                    f"(Best {best_val_loss:.4f}|{best_val_acc:4f}|{best_val_F1})\n Best Test F1 - {bestF1_testF1}")

                if validation_loss > best_val_loss and epoch > best_val_loss_epoch + early_stop_after:
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

    @torch.no_grad()
    def validate(self, model: torch.nn.Module, lossfunction: _Loss, dev_iter: Iterator, config: dict, verbose=False,
                 log_results=True):
        train_flag = model.training
        model.eval()

        total_batches = len(dev_iter.data()) // dev_iter.batch_size
        if verbose:
            pbar = tqdm(total=total_batches)
        if log_results:
            csvf, workbook, worksheet, writer = self.init_result_logging()
        examples_so_far = 0
        dev_loss = 0
        total_correct = 0
        total_correct_per_level = Counter()
        total_per_level = defaultdict(lambda: 0)

        total_labels = []
        total_preds = []
        for i, batch in enumerate(dev_iter):
            pred_logits, attention = model(batch)

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
                id_s = batch.tweet_id
                text_s = [' '.join(self.tokenizer.convert_ids_to_tokens(batch.text[i].cpu().numpy())) for i in
                          range(batch.text.shape[0])]
                pred_s = list(argmaxpreds.cpu().numpy())
                target_s = list(batch.stance_label.cpu().numpy())
                correct_s = list((argmaxpreds == batch.stance_label).cpu().numpy())
                prob_s = [f"{x:.2f}" for x in
                          list(maxpreds.cpu().detach().numpy())]

                assert len(text_s) == len(pred_s) == len(correct_s) == len(
                    target_s) == len(prob_s)
                global row
                for i in range(len(text_s)):
                    writer.writerow([id_s[i], correct_s[i],
                                     map_stance_label_to_s[target_s[i]],
                                     map_stance_label_to_s[pred_s[i]],
                                     prob_s[i],
                                     text_s[i]])
                    res = [id_s[i], correct_s[i],
                           map_stance_label_to_s[target_s[i]],
                           map_stance_label_to_s[pred_s[i]],
                           prob_s[i]]

                    for col in range(len(res)):
                        worksheet.write(row, col, str(res[col]))

                    att_contexts = text_s[i].split()
                    att_sum_vec = attention[i].sum(0)

                    att_vector = (att_sum_vec / att_sum_vec.max() * Text_Framework_Seq.COLOR_RESOLUTION).int()
                    for col in range(len(res), len(res) + len(att_contexts)):
                        j = col - len(res)
                        worksheet.write(row, col, att_sum_vec[j].item())
                    row += 1
                    for col in range(len(res), len(res) + len(att_contexts)):
                        j = col - len(res)
                        selectedc = Text_Framework_Seq.colors[max(att_vector[j].item() - 1, 0)].get_hex_l()
                        opts = {'bg_color': selectedc}
                        myformat = workbook.add_format(opts)
                        worksheet.write(row, col, att_contexts[j], myformat)

                    row += 1
                    for filt in range(attention[i].shape[0]):
                        att_vector = (attention[i][filt] / attention[i][
                            filt].max() * Text_Framework_Seq.COLOR_RESOLUTION).int()
                        for col in range(len(res), len(res) + len(att_contexts)):
                            j = col - len(res)
                            selectedc = Text_Framework_Seq.colors[max(att_vector[j].item() - 1, 0)].get_hex_l()
                            opts = {'bg_color': selectedc}
                            myformat = workbook.add_format(opts)
                            worksheet.write(row, col, att_contexts[j], myformat)
                        row += 1

        loss, acc = dev_loss / total_batches, total_correct / examples_so_far
        total_acc_per_level = {depth: total_correct_per_level.get(depth, 0) / total for depth, total in
                               total_per_level.items()}
        F1 = metrics.f1_score(total_labels, total_preds, average="macro")
        allF1s = metrics.f1_score(total_labels, total_preds, average=None).tolist()
        if log_results:
            self.finalize_results_logging(csvf, workbook, F1, loss)
        if train_flag:
            model.train()
        return loss, acc, total_acc_per_level, F1, allF1s

    def finalize_results_logging(self, csvf, workbook, score, loss):
        csvf.close()
        workbook.close()

        os.rename(self.TMP_FNAME + ".tsv", f"introspection/introspection"
        f"_{str(self.__class__)}_A{score:.6f}_L{loss:.6f}_{socket.gethostname()}.tsv", )
        os.rename(self.TMP_FNAME + ".xlsx", f"introspection/introspection"
        f"_{str(self.__class__)}_A{score:.6f}_L{loss:.6f}_{socket.gethostname()}.xlsx", )

    RESULT_HEADER = ["ID",
                     "Correct",
                     "Ground truth",
                     "Prediction",
                     "Confidence",
                     "Text"]

    def init_result_logging(self):
        self.TMP_FNAME = f"introspection/fulltext_{get_timestamp()}_{socket.gethostname()}"
        csvf = open(f"{self.TMP_FNAME}.tsv", mode="w")
        writer = csv.writer(csvf, delimiter='\t')
        writer.writerow(self.RESULT_HEADER)
        global row
        row = 0
        workbook = xlsxwriter.Workbook(f"{self.TMP_FNAME}.xlsx")
        worksheet = workbook.add_worksheet()
        for col in range(len(self.RESULT_HEADER)):
            worksheet.write(row, col, self.RESULT_HEADER[col])
        row += 1
        return csvf, workbook, worksheet, writer
