import logging
import math
import time
from itertools import chain

import torch
from torch.nn.modules.loss import _Loss
from torchtext.data import BucketIterator, Iterator
from tqdm import tqdm

from RumourEvalDataset_Branches import RumourEval2019Dataset_Branches
from modelutils import glorot_param_init, disable_gradients
from task_A.frameworks.base_framework import Base_Framework
#from tflogger import TBLogger
from utils import count_parameters, get_timestamp


# tblogging = TBLogger('./logs')


class Text_Feature_Framework(Base_Framework):
    def __init__(self, config: dict):
        super().__init__(config)

    def train(self, modelfunc, seed=42):
        if seed is not None:
            torch.manual_seed(seed)

        config = self.config
        fields = RumourEval2019Dataset_Branches.prepare_fields_for_text(self.config["hyperparameters"]["sep_token"])
        train_data, train_fields = self.build_dataset(config["train_data"], fields)
        dev_data, dev_fields = self.build_dataset(config["dev_data"], fields)

        torch.manual_seed(42)

        # No need to build vocab for baseline
        # but fo future work I wrote RumourEval2019Dataset that
        # requires vocab to be build
        build_vocab = lambda field, *data: RumourEval2019Dataset_Branches.build_vocab(field,
                                                                                      self.config["hyperparameters"][
                                                                                          "sep_token"],
                                                                                      *data,
                                                                                      vectors=config["embeddings"],
                                                                                      vectors_cache=config[
                                                                                          "vector_cache"])
        # FIXME: add dev vocab to model's vocab
        # 14833 together
        # 11809 words in train
        # 6450 words in dev
        build_vocab(train_fields['spacy_processed_text'], train_data, dev_data)
        self.vocab = train_fields['spacy_processed_text'].vocab

        device = torch.device("cuda:0" if config['cuda'] and
                                          torch.cuda.is_available() else "cpu")

        create_iter = lambda data: BucketIterator(data, sort_key=lambda x: len(x.stance_labels), sort=True,
                                                  batch_size=config["hyperparameters"]["batch_size"], repeat=False,
                                                  device=device)
        train_iter = create_iter(train_data)
        dev_iter = create_iter(dev_data)

        logging.info(f"Train examples: {len(train_data.examples)}\nValidation examples: {len(dev_data.examples)}")

        model = modelfunc(self.config, vocab=train_fields['spacy_processed_text'].vocab,
                          textmodel=torch.load(
                              "saved/checkpoint_<class 'task_A.frameworks.text_framework.Text_Framework'>_ACC_0.71184_2018-12-11_21:04.pt")) \
            .to(device)

        glorot_param_init(model)
        logging.info(f"Model has {count_parameters(model)} trainable parameters.")
        logging.info(f"Manual seed {torch.initial_seed()}")
        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
        #                             lr=config["hyperparameters"]["learning_rate"])

        optimizer_for_hcrafted = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                         chain(
                                                             model.baseline_feature_extractor.parameters(),
                                                             model.final_layer.parameters())),
                                                  lr=0.00008,
                                                  betas=[0.9, 0.999], eps=1e-8)
        # optimizer_for_e2e = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad,
        #            chain(model.textselfatt_feature_extractor.parameters(), model.final_layer.parameters())),
        #     lr=0.00001)
        # final_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.final_layer.parameters()),
        #                                    lr=0.0001,
        #                                    betas=[0.9, 0.999], eps=1e-8)

        lossfunction = torch.nn.CrossEntropyLoss()
        start_time = time.time()
        try:
            best_val_loss = math.inf
            best_val_acc = 0
            disabled = False
            p1 = p2 = p3 = False
            for epoch in range(config["hyperparameters"]["epochs"]):
                # if epoch < 80:
                #     if not p1:
                #         logging.info("#" * 96)
                #         logging.info("Training phase 1")
                #         p1 = True
                #     selected_features = "text"
                #     optimizer = optimizer_for_e2e
                # elif epoch < 80:
                #     torch.save(model, f"saved/submodel_1_2_pretrained_{get_timestamp()}.pt")
                #     exit()
                #     if not p2:
                #         logging.info("#" * 96)
                #         logging.info("Training phase 2")
                #         p2 = True
                #     selected_features = "hand"
                #     #optimizer = optimizer_for_hcrafted
                # else:
                #     if not p3:
                #         logging.info("#" * 96)
                #         logging.info("Training phase 3")
                #         p3 = True
                #     #optimizer = final_optimizer
                #     if not disabled:
                #         logging.info("Disabling gradients")
                #         disable_gradients(model.baseline_feature_extractor)
                #         disable_gradients(model.textselfatt_feature_extractor)
                #         disabled = True
                #    selected_features = "all"

                self.run_epoch(model, lossfunction, optimizer_for_hcrafted, train_iter, config, None)

                train_loss, train_acc = self.validate(model, lossfunction, train_iter, config,
                                                      None)

                logging.info(
                    f"Epoch {epoch}, Training loss|acc: {train_loss:.6f}|{train_acc:.6f}")
                validation_loss, validation_acc = self.validate(model, lossfunction, dev_iter, config,
                                                                selected_features)
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                if validation_acc > best_val_acc:
                    best_val_acc = validation_acc
                logging.info(
                    f"Epoch {epoch}, Validation loss|acc: {validation_loss:.6f}|{validation_acc:.6f} - (Best {best_val_loss:.4f}|{best_val_acc:4f})")
        except KeyboardInterrupt:
            logging.info('-' * 120)
            logging.info('Exit from training early.')
        finally:
            logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')

    # def run_epoch(self, model, lossfunction, optimizer, train_iter, config, verbose=False):
    #     total_batches = len(train_iter.data()) // train_iter.batch_size
    #     if verbose:
    #         pbar = tqdm(total=total_batches)
    #     examples_so_far = 0
    #     train_loss = 0
    #     total_correct = 0
    #     for i, batch in enumerate(train_iter):
    #         optimizer.zero_grad()
    #         pred_logits = model(batch)
    #
    #         pred_logits = pred_logits.view((-1, 4))
    #         labels = batch.stance_labels.view(-1)
    #         mask = labels > -1
    #         masked_preds, masked_labels = pred_logits[mask, :], labels[mask]
    #         loss = lossfunction(masked_preds, masked_labels)
    #
    #         loss.backward()
    #         if model.get_encoder() is not None:
    #             torch.nn.utils.clip_grad_norm_(model.get_encoder().parameters(), config["hyperparameters"]["RNN_clip"])
    #         optimizer.step()
    #
    #         total_correct += self.calculate_correct(masked_preds, masked_labels)
    #         examples_so_far += 1
    #         train_loss += loss.item()
    #         if verbose:
    #             pbar.set_description(
    #                 f"Train loss: {train_loss / (i + 1):.4f}, Train acc: {total_correct / examples_so_far:.4f}")
    #             pbar.update(1)
    #     return train_loss / train_iter.batch_size, total_correct / examples_so_far

    def run_epoch(self, model, lossfunction, optimizer, train_iter, config, selected_features, verbose=False):
        global step
        total_batches = len(train_iter.data()) // train_iter.batch_size
        if verbose:
            pbar = tqdm(total=total_batches)
        examples_so_far = 0
        train_loss = 0
        total_correct = 0
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            pred_logits = model(batch, selected_features)

            pred_logits = pred_logits.view((-1, 4))
            labels = batch.stance_labels.view(-1)
            mask = labels > -1
            masked_preds, masked_labels = pred_logits[mask, :], labels[mask]
            loss = lossfunction(masked_preds, masked_labels)

            loss.backward()
            if model.get_encoder() is not None:
                torch.nn.utils.clip_grad_norm_(model.get_encoder().parameters(), config["hyperparameters"]["RNN_clip"])
            optimizer.step()

            total_correct += self.calculate_correct(masked_preds, masked_labels)
            examples_so_far += 1
            train_loss += loss.item()
            if verbose:
                pbar.update(1)
            if step % 500 == 0:
                self.tblog_model_parameters(model, step)
            step += 1
        return train_loss / total_batches, total_correct / examples_so_far

    def tblog_model_parameters(self, model, step):
        for tag, value in model.named_parameters():
            if value.requires_grad:
                tag = tag.replace('.', '/')
                tblogging.histo_summary(tag, value.data.cpu().numpy(), step)
                if value.grad is not None:
                    tblogging.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step)

    def validate(self, model: torch.nn.Module, lossfunction: _Loss, dev_iter: Iterator, config: dict, selected_features,
                 verbose=False):
        train_flag = model.training
        model.eval()

        total_batches = len(dev_iter.data()) // dev_iter.batch_size
        if verbose:
            pbar = tqdm(total=total_batches)
        examples_so_far = 0
        dev_loss = 0
        total_correct = 0
        for i, batch in enumerate(dev_iter):
            pred_logits = model(batch, selected_features)

            pred_logits = pred_logits.view((-1, 4))
            labels = batch.stance_labels.view(-1)
            mask = labels > -1
            masked_preds, masked_labels = pred_logits[mask, :], labels[mask]
            loss = lossfunction(masked_preds, masked_labels)

            total_correct += self.calculate_correct(masked_preds, masked_labels)
            examples_so_far += len(masked_labels)
            dev_loss += loss.item()
            if verbose:
                pbar.set_description(
                    f"dev loss: {dev_loss / (i + 1):.4f}, dev acc: {total_correct / examples_so_far:.4f}")
                pbar.update(1)
        if train_flag:
            model.train()
        return dev_loss / total_batches, total_correct / examples_so_far

    def calculate_correct(self, pred_logits: torch.Tensor, labels: torch.Tensor):
        preds = torch.argmax(pred_logits, dim=1)
        return torch.sum(preds == labels).item()
