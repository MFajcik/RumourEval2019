import csv

import torch
import torch.nn.functional as F
import xlsxwriter
from colour import Color
from torch.nn.modules.loss import _Loss
from torchtext.data import Iterator
from tqdm import tqdm

from task_A.frameworks.base_framework import Base_Framework
from utils import totext


def frobenius_norm(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1), 1).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')


class Text_Framework(Base_Framework):
    RESULT_HEADER = ["Correct",
                     "Ground truth",
                     "Prediction",
                     "Confidence",
                     "Text"]
    COLOR_GRADIENT_FROM = Color("#ffffff")
    COLOR_GRADIENT_TO = Color("#002eff")
    COLOR_RESOLUTION = 1000
    colors = list(COLOR_GRADIENT_FROM.range_to(COLOR_GRADIENT_TO, COLOR_RESOLUTION))

    def __init__(self, config: dict):
        super().__init__(config)
        self.save_treshold = 0.71
        self.I = torch.eye(config['hyperparameters']['ATTENTION_hops'], requires_grad=False,
                           device=torch.device("cuda:0" if config['cuda'] and
                                                           torch.cuda.is_available() else "cpu")) \
            .unsqueeze(0)

    def run_epoch(self, model, lossfunction, optimizer, train_iter, config, verbose=False):
        total_batches = len(train_iter.data()) // train_iter.batch_size
        if verbose:
            pbar = tqdm(total=total_batches)
        examples_so_far = 0
        train_loss = 0
        total_correct = 0
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            pred_logits, attention = model(batch)

            pred_logits = pred_logits.view((-1, 4))
            labels = batch.stance_labels.view(-1)
            mask = labels > -1
            masked_preds, masked_labels = pred_logits[mask, :], labels[mask]
            loss = lossfunction(masked_preds, masked_labels)

            if config["hyperparameters"]["cov_penalization"] > 1e-10:
                attentionT = torch.transpose(attention, 1, 2).contiguous()
                # We index I because of the last batch, where we need less identity matrices, than batch_size
                extra_loss = frobenius_norm(torch.bmm(attention, attentionT) - self.I.expand(attention.size(0),
                                                                                             config['hyperparameters'][
                                                                                                 'ATTENTION_hops'],
                                                                                             config['hyperparameters'][
                                                                                                 'ATTENTION_hops']))
                loss += config["hyperparameters"]["cov_penalization"] * extra_loss

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
        return train_loss / train_iter.batch_size, total_correct / examples_so_far

    def validate(self, model: torch.nn.Module, lossfunction: _Loss, dev_iter: Iterator, config: dict, verbose=False,
                 log_results=True, vocab=None):
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

            pred_logits = pred_logits.view((-1, 4))
            labels = batch.stance_labels.view(-1)
            mask = labels > -1
            masked_logits, masked_labels = pred_logits[mask, :], labels[mask]
            loss = lossfunction(masked_logits, masked_labels)

            total_correct += self.calculate_correct(masked_logits, masked_labels)
            examples_so_far += len(masked_labels)
            dev_loss += loss.item()
            if verbose:
                pbar.set_description(
                    f"dev loss: {dev_loss / (i + 1):.4f}, dev acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)
            if log_results:
                maxpreds, argmaxpreds = torch.max(F.softmax(masked_logits, -1), dim=1)
                inp = model.prepare_inp(batch)
                text_s = totext(inp.view(-1, inp.shape[-1])[mask], self.vocab)
                pred_s = list(argmaxpreds.cpu().numpy())
                target_s = list(masked_labels.cpu().numpy())
                correct_s = list((argmaxpreds == masked_labels).cpu().numpy())
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
        return dev_loss / dev_iter.batch_size, total_correct / examples_so_far

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
