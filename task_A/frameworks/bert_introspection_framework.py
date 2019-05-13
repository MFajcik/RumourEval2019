import fileinput
import logging
import sys
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from past.builtins import raw_input
from pytorch_pretrained_bert import BertTokenizer
from sklearn import metrics
from torchtext.data import BucketIterator
from tqdm import tqdm

from task_A.datasets.RumourEvalDataset_BERT import RumourEval2019Dataset_BERTTriplets
from task_A.frameworks.bert_framework import BERT_Framework
from utils.utils import map_stance_label_to_s
from utils.utils import count_parameters

class BERT_Introspection_Framework(BERT_Framework):
    def __init__(self, config: dict):
        super().__init__(config)
        self.init_tokenizer()

        import seaborn
        seaborn.set(font_scale=0.7)

    def init_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", cache_dir="./.BERTcache",
                                                       do_lower_case=True)

    def fit(self, modelfunc):
        config = self.config

        fields = RumourEval2019Dataset_BERTTriplets.prepare_fields_for_text()
        train_data = RumourEval2019Dataset_BERTTriplets(config["train_data"], fields, self.tokenizer,
                                                        max_length=config["hyperparameters"]["max_length"])
        dev_data = RumourEval2019Dataset_BERTTriplets(config["dev_data"], fields, self.tokenizer,
                                                      max_length=config["hyperparameters"]["max_length"])

        # torch.manual_seed(5246727901370826861 & ((1 << 63) - 1))
        # torch.manual_seed(40)

        # 84.1077

        device = torch.device("cuda:0" if config['cuda'] and
                                          torch.cuda.is_available() else "cpu")

        train_iter = BucketIterator(train_data,
                                    sort_key=lambda x: -len(x.text), sort=True,
                                    # shuffle=True,
                                    batch_size=config["hyperparameters"]["batch_size"],
                                    repeat=False,
                                    device=device)
        dev_iter = BucketIterator(dev_data,  # sort_key=lambda x: -len(x.text), sort=True,
                                  shuffle=True,
                                  batch_size=config["hyperparameters"]["batch_size"], repeat=False,
                                  device=device)

        logging.info(f"Train examples: {len(train_data.examples)}\nValidation examples: {len(dev_data.examples)}")
        # model = modelfunc.from_pretrained("bert-base-uncased", cache_dir="./.BERTcache").to(device)
        pretrained_model = torch.load(
            "saved/ensemble/BIG_checkpoint_<class 'task_A.frameworks.bert_framework_"
            "hyperparam_tuning.BERT_Framework_Hyperparamopt'>_"
            "F1_0.58229_L_0.6772208201254635_2019-01-27_18:12_pcknot4.pt").to(
            device)
        model = modelfunc.from_pretrained("bert-large-uncased", cache_dir="./.BERTcache",
                                          state_dict=pretrained_model.state_dict()
                                          ).to(device)
        logging.info(f"Model has {count_parameters(model)} trainable parameters.")
        logging.info(f"Manual seed {torch.initial_seed()}")

        start_time = time.time()
        try:
            l, acc, perlevelF1, F1 = self.introspect(model, dev_iter, config,
                                                     log_results=True)
            logging.info(
                f"loss|acc|F1: {l:.6f}|{acc:.6f}|{F1:.6f}")
        except KeyboardInterrupt:
            logging.info('-' * 120)
            logging.info('Exit from training early.')
        finally:
            logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')

    def draw(self, data, x, y, ax, cmap="Blues"):
        import seaborn
        seaborn.heatmap(data,
                        xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,
                        cbar=False, ax=ax, cmap=cmap)

    def introspect(self, model: torch.nn.Module, dev_iter: BucketIterator, config: dict, verbose=False,
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

            text_s = [' '.join(self.tokenizer.convert_ids_to_tokens(batch.text[i].cpu().numpy())) for i in
                      range(batch.text.shape[0])]
            print(text_s[0])
            # 0: "support",
            # 1: "comment",
            # 2: "deny",
            # 3: "query"
            branch_levels = [id.split(".", 1)[-1] for id in batch.branch_id]
            for branch_depth in branch_levels: total_per_level[branch_depth] += 1
            correct, correct_per_level = self.calculate_correct(pred_logits, batch.stance_label, levels=branch_levels)
            total_correct += correct
            total_correct_per_level += correct_per_level

            prefix = "exccomment"
            # works only with batch size 1
            if batch.stance_label != 1 and correct == True:
                c = raw_input('please confirm ')
                if c == "y":
                    self.generate_attention_images(batch, model, pred_logits, prefix)
                    sys.exit()

            examples_so_far += len(batch.stance_label)
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

    def generate_attention_images(self, batch, model, pred_logits, prefix):
        words = [self.tokenizer.convert_ids_to_tokens(batch.text[i].cpu().numpy()) for i in
                 range(batch.text.shape[0])]

        import matplotlib.pyplot as plt
        for layer in range(24):
            subplots_per_plot = 1
            fig, axs = plt.subplots(1, subplots_per_plot, figsize=(20, 10))
            print("Encoder Layer", layer + 1)
            for b in range(pred_logits.shape[0]):
                for h in range(16):
                    sent = words[b]
                    self.draw(model.bert.encoder.all_attention_probs[layer][b, h].cpu().data,
                              sent, sent, ax=axs)
                    plt.savefig(f'images/{prefix}_L_{layer}_H{h * subplots_per_plot + 1}_{h * subplots_per_plot + subplots_per_plot}.png', dpi=400)
        sys.exit()
