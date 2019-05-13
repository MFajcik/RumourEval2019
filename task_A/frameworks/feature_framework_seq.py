import csv
import logging
import math
import time

import torch
import torch.nn.functional as F
import xlsxwriter
from colour import Color
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
from torchtext.data import Iterator, BucketIterator
from tqdm import tqdm

from task_A.datasets.RumourEvalDataset_Seq import RumourEval2019Dataset_Seq
from task_A.frameworks.base_framework import Base_Framework
from task_A.frameworks.text_framework_branch import Text_Framework
from utils import totext, count_parameters, get_timestamp


class Feature_Framework_Seq(Base_Framework):
    RESULT_HEADER = ["Correct",
                     "Ground truth",
                     "Prediction",
                     "Confidence",
                     "Text"]

    def __init__(self, config: dict):
        super().__init__(config)
        self.save_treshold = 0.90
        self.I = torch.eye(config['hyperparameters']['ATTENTION_hops'], requires_grad=False,
                           device=torch.device("cuda:0" if config['cuda'] and
                                                           torch.cuda.is_available() else "cpu")) \
            .unsqueeze(0)

    def fit(self, modelfunc):
        config = self.config

        fields = RumourEval2019Dataset_Seq.prepare_fields()
        train_data = RumourEval2019Dataset_Seq(config["train_data"], fields)
        dev_data = RumourEval2019Dataset_Seq(config["dev_data"], fields)
        test_data = RumourEval2019Dataset_Seq(config["test_data"], fields)

        fields = {k: v for k, v in fields}

        # FIXME: add dev vocab to model's vocab
        # 14833 together
        # 11809 words in train
        # 6450 words in dev
        fields['spacy_processed_text'].build_vocab(train_data, dev_data, test_data, vectors=config["embeddings"],
                                                   vectors_cache=config["vector_cache"])
        self.vocab = fields['spacy_processed_text'].vocab

        device = torch.device("cuda:0" if config['cuda'] and
                                          torch.cuda.is_available() else "cpu")

        create_iter = lambda data: BucketIterator(data, sort_key=lambda x: -len(x.spacy_processed_text), sort=True,
                                                  # shuffle=True,
                                                  batch_size=config["hyperparameters"]["batch_size"], repeat=False,
                                                  device=device)

        train_iter = create_iter(train_data)
        dev_iter = create_iter(dev_data)
        test_iter = create_iter(test_data)

        logging.info(f"Train examples: {len(train_data.examples)}\nValidation examples: {len(dev_data.examples)}")

        model = modelfunc(self.config, vocab=self.vocab).to(device)

        logging.info(f"Model has {count_parameters(model)} trainable parameters.")
        logging.info(f"Manual seed {torch.initial_seed()}")
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=config["hyperparameters"]["learning_rate"])
        lossfunction = torch.nn.CrossEntropyLoss()
        start_time = time.time()
        try:
            best_val_loss = math.inf
            best_val_acc = 0
            for epoch in range(config["hyperparameters"]["epochs"]):
                train_loss, train_acc = self.train(model, lossfunction, optimizer, train_iter, config)
                validation_loss, validation_acc = self.validate(model, lossfunction, dev_iter, config)
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                if validation_acc > best_val_acc:
                    best_val_acc = validation_acc
                logging.info(
                    f"Epoch {epoch}, Validation loss|acc: {validation_loss:.6f}|{validation_acc:.6f} - (Best {best_val_loss:.4f}|{best_val_acc:4f})")
                if validation_acc > self.save_treshold:
                    torch.save(model.to(torch.device("cpu")),
                               f"saved/checkpoint_{str(self.__class__)}_ACC_{validation_acc:.5f}_{get_timestamp()}.pt")
                    model.to(torch.device(device))
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
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            pred_logits, attention = model(batch)

            loss = lossfunction(pred_logits, batch.stance_label)
            loss.backward()
            if model.encoder is not None:
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), config["hyperparameters"]["RNN_clip"])
            optimizer.step()

            total_correct += self.calculate_correct(pred_logits, batch.stance_label)
            examples_so_far += 1
            train_loss += loss.item()
            if verbose:
                pbar.set_description(
                    f"Train loss: {train_loss / (i + 1):.4f}, Train acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)
        return train_loss / total_batches, total_correct / examples_so_far

    def validate(self, model: torch.nn.Module, lossfunction: _Loss, dev_iter: Iterator, config: dict, verbose=False,
                 log_results=False, vocab=None):
        train_flag = model.training
        model.eval()
        if log_results:
            csvf, workbook, worksheet, writer = self.init_result_logging()

        total_batches = len(dev_iter.data()) // dev_iter.batch_size
        if verbose:
            pbar = tqdm(total=total_batches)
        examples_so_far = 0
        dev_loss = 0
        total_correct = 0
        for i, batch in enumerate(dev_iter):
            pred_logits, attention = model(batch)

            loss = lossfunction(pred_logits, batch.stance_label)

            total_correct += self.calculate_correct(pred_logits, batch.stance_label)
            examples_so_far += len(batch.stance_label)
            dev_loss += loss.item()

            if verbose:
                pbar.set_description(
                    f"dev loss: {dev_loss / (i + 1):.4f}, dev acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)
            if log_results:
                maxpreds, argmaxpreds = torch.max(F.softmax(pred_logits, -1), dim=1)
                inp = model.prepare_inp(batch)
                text_s = totext(inp.view(-1, inp.shape[-1]), self.vocab)
                pred_s = list(argmaxpreds.cpu().numpy())
                target_s = list(batch.stance_label.cpu().numpy())
                correct_s = list((argmaxpreds == batch.stance_label).cpu().numpy())
                prob_s = [f"{x:.2f}" for x in
                          list(maxpreds.cpu().detach().numpy())]

                assert len(text_s) == len(pred_s) == len(correct_s) == len(
                    target_s) == len(prob_s)
                for i in range(len(text_s)):
                    writer.writerow([correct_s[i],
                                     target_s[i],
                                     pred_s[i],
                                     prob_s[i],
                                     text_s[i]])
                    # res = [correct_s[i],
                    #        category_s[i],
                    #        prob_s[i],
                    #        histogram[target_s[i]].item(),
                    #        root_s[i],
                    #        target_s[i],
                    #        pred_s[i]]
                    # for col in range(len(res)):
                    #     worksheet.write(row, col, str(res[col]))
                    #
                    # att_contexts = text_s[i].split()
                    # att_sum_vec = attention[i].sum(0)
                    #
                    # att_vector = (att_sum_vec / att_sum_vec.max() * COLOR_RESOLUTION).int()
                    # for col in range(len(res), len(res) + len(att_contexts)):
                    #     j = col - len(res)
                    #     worksheet.write(row, col, att_sum_vec[j].item())
                    # row += 1
                    # for col in range(len(res), len(res) + len(att_contexts)):
                    #     j = col - len(res)
                    #     selectedc = colors[max(att_vector[j].item() - 1, 0)].get_hex_l()
                    #     opts = {'bg_color': selectedc}
                    #     myformat = workbook.add_format(opts)
                    #     worksheet.write(row, col, att_contexts[j], myformat)
                    #
                    # row += 1
                    # for filt in range(attention[i].shape[0]):
                    #     att_vector = (attention[i][filt] / attention[i][filt].max() * COLOR_RESOLUTION).int()
                    #     for col in range(len(res), len(res) + len(att_contexts)):
                    #         j = col - len(res)
                    #         selectedc = colors[max(att_vector[j].item() - 1, 0)].get_hex_l()
                    #         opts = {'bg_color': selectedc}
                    #         myformat = workbook.add_format(opts)
                    #         worksheet.write(row, col, att_contexts[j], myformat)
                    #     row += 1
        if train_flag:
            model.train()
        if log_results:
            self.finalize_results_logging(csvf, workbook)
        return dev_loss / total_batches, total_correct / examples_so_far

    def finalize_results_logging(self, csvf, workbook):
        csvf.close()
        workbook.close()

    def init_result_logging(self):
        csvf = open(f"introspection.tsv", mode="w")
        writer = csv.writer(csvf, delimiter='\t')
        writer.writerow(Text_Framework.RESULT_HEADER)
        global row
        row = 0
        workbook = xlsxwriter.Workbook(f"introspection.xlsx")
        worksheet = workbook.add_worksheet()
        for col in range(len(Text_Framework.RESULT_HEADER)):
            worksheet.write(row, col, Text_Framework.RESULT_HEADER[col])
        row += 1
        return csvf, workbook, worksheet, writer

    def calculate_correct(self, pred_logits: torch.Tensor, labels: torch.Tensor):
        argpreds = torch.argmax(pred_logits, dim=1)
        return torch.sum(argpreds == labels).item()
