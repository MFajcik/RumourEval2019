import json
import logging
from collections import Iterator

import torch
from pytorch_pretrained_bert import BertTokenizer
from torchtext.data import BucketIterator
from tqdm import tqdm

from task_A.datasets.RumourEvalDataset_BERT import RumourEval2019Dataset_BERTTriplets
from task_A.frameworks.base_framework import Base_Framework
from utils.utils import map_stance_label_to_s


class BERT_Framework_Hyperparamopt_WITHSANITY(Base_Framework):
    def __init__(self, config: dict):
        super().__init__(config)
        self.save_treshold = 0.53
        self.modeltype = config["variant"]
        self.tokenizer = BertTokenizer.from_pretrained(self.modeltype, cache_dir="./.BERTcache",
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
        updated = False
        for i, batch in enumerate(train_iter):
            updated = False
            pred_logits = model(batch)

            loss = lossfunction(pred_logits, batch.stance_label) / update_ratio
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
            model.fit()
        logging.info(f"Writing results into {fname}")

    def train(self, modelfunc):
        config = self.config

        fields = RumourEval2019Dataset_BERTTriplets.prepare_fields_for_text()
        train_data = RumourEval2019Dataset_BERTTriplets(config["train_data"], fields, self.tokenizer,
                                                        max_length=config["hyperparameters"]["max_length"])
        train_data, sanity_check_data = train_data.split(split_ratio=0.9, stratified=True, strata_field='stance_label')
        dev_data = RumourEval2019Dataset_BERTTriplets(config["dev_data"], fields, self.tokenizer,
                                                      max_length=config["hyperparameters"]["max_length"])

        device = torch.device("cuda:0" if config['cuda'] and
                                          torch.cuda.is_available() else "cpu")

        train_iter, sanity_iter = BucketIterator.splits((train_data, sanity_check_data),
                                                        sort_key=lambda x: -len(x.text), sort=True,
                                                        # shuffle=True,
                                                        batch_size=config["hyperparameters"]["batch_size"],
                                                        repeat=False,
                                                        device=device)
        dev_iter = BucketIterator(dev_data, sort_key=lambda x: -len(x.text), sort=True,
                                  # shuffle=True,
                                  batch_size=config["hyperparameters"]["batch_size"], repeat=False,
                                  device=device)
        # train_iter = create_iter(train_data)
        # dev_iter = create_iter(dev_data)

        logging.info(f"Train examples: {len(train_data.examples)}\n"
                     f"Sanity-check examples: {len(sanity_check_data.examples)}\n"
                     f"Validation examples: {len(dev_data.examples)}")

        # bert-base-uncased
        # bert-large-uncased,
        # bert-base-multilingual-cased
        # pretrained_model = torch.load(
        #     "saved/checkpoint_<class 'task_A.frameworks.bert_framework.BERT_Framework'>_ACC_0.83704_2019-01-10_12:14.pt").to(
        #     device)
        # model = modelfunc.from_pretrained("bert-base-uncased", cache_dir="./.BERTcache",
        #                                   state_dict=pretrained_model.state_dict()
        #                                   ).to(device)
        # pretrained_model = None

        # ~ 32 mins per one training
        # search_space = {"learning_rate": hp.choice("learning_rate", [1e-06, 5e-06, 1e-05, 5e-5, 5e-7]),
        #                 "true_batch_size": hp.choice("true_batch_size", [8, 16, 32, 64]),
        #                 "max_length": hp.choice("max_length", [50, 100, 150, 180, 200, 250, 512]),
        #                 "hidden_dropout_prob": hp.choice("hidden_dropout_prob", [0., 0.1, 0.3, 0.4, 0.5])
        #                 }
        #
        #
        # trials = Trials()
        # ntrials = 50
        # global counter
        # counter = 0
        # def obj(params):
        #     global counter
        #     print(f"called {counter}")
        #     counter+=1
        #     output = {'loss': random.uniform(0.5, 99), 'status': STATUS_OK}
        #     return output
        #
        # best = fmin(obj,
        #             space=search_space,
        #             algo=tpe.suggest,
        #             max_evals=ntrials,
        #             trials=trials)
        # logging.info(best)
        # bp = trials.best_trial['result']['Params']
        #
        # try:
        #     f = open('output/trials_BERT.txt', "wb")
        #     pickle.dump(trials, f)
        #     f.close()
        #
        #     filename = 'output/bestparams_BERT.txt'
        #     f = open(filename, "wb")
        #     pickle.dump(bp, f)
        #     f.close()
        # except Exception as e:
        #     logging.error("An exception caused params were not saved!")
        #     logging.error(e)
        #

        ##FIXME: NOT HERE!!
        hyperparamopt = False
        parameter_to_optimize = None  ##"true_batch_size"

        if not hyperparamopt:
            for i in range(200):
                torch.manual_seed(random.randint(1, 1e8))
                self.run_training(config, dev_iter, sanity_iter, device, modelfunc, train_data, train_iter)
        else:
            params = {"learning_rate": [1e-06, 9e-7, 8e-7, 7e-7, 2e-06],
                      "true_batch_size": [64, 80, 128],
                      "max_length": [512, 200, 220, 250],
                      "hidden_dropout_prob": [0., 0.1, 0.3, 0.4, 0.5]}
            trials = 6

            logging.info(f"Optimizing {parameter_to_optimize}")

            def test():
                best_F = random.uniform(0.5, 99)
                best_loss = random.uniform(0.5, 99)
                F = best_F / 2
                loss = best_loss / 2
                return {
                    "best_loss": best_loss,
                    "best_F1": best_F,
                    "bestF1_loss": loss,
                    "bestloss_F1": F
                }

            results = []
            for par in params[parameter_to_optimize]:
                config["hyperparameters"][parameter_to_optimize] = par
                logging.info(f"{parameter_to_optimize}:{par}")
                for i in range(trials):
                    torch.manual_seed(random.randint(1, 1e6))
                    results.append(self.run_training(config, dev_iter, device, modelfunc, train_data, train_iter))
                    # results.append(test())
            logging.info("Results:")
            for pari, par in enumerate(params[parameter_to_optimize]):
                logging.info(f"{parameter_to_optimize}:{par}")
                for i in range(trials):
                    logging.info(json.dumps(results[pari * trials + i]))

                logging.info("*" * 20 + "AVG" + "*" * 20)
                avg = Counter(results[0])
                for i in range(1, trials): avg += Counter(results[pari * trials + i])
                for key in avg:
                    avg[key] /= trials
                logging.info(json.dumps(avg))
                logging.info("*" * 20 + "AVG ends" + "*" * 20)

    def run_training(self, config, dev_iter, sanity_iter, device, modelfunc, train_data, train_iter):
        logger = logging.getLogger()
        logger.disabled = True

        self.tokenizer = BertTokenizer.from_pretrained(self.modeltype, cache_dir="./.BERTcache",
                                                       do_lower_case=True)
        model = modelfunc.from_pretrained(self.modeltype, cache_dir="./.BERTcache").to(device)
        weights = SelfAtt_BertTokenizing_Framework.get_class_weights(train_data.examples, "stance_label", 4,
                                                                     min_fraction=1)
        logger.disabled = False
        model.reinit(config)
        logging.info(f"Model has {count_parameters(model)} trainable parameters.")
        logging.info(f"Manual seed {torch.initial_seed()}")
        optimizer = BertAdam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=config["hyperparameters"]["learning_rate"])
        logging.info("class weights")
        logging.info(f"{str(weights.numpy().tolist())}")
        lossfunction = torch.nn.CrossEntropyLoss(weight=weights.to(device))
        best_val_loss = math.inf
        best_val_acc = 0
        best_val_F1 = 0
        best_F1_loss, best_loss_F1 = 0, 0
        start_time = time.time()
        try:

            # self.predict("answer_BERT_textnsource.json", model, dev_iter)
            best_val_los_epoch = -1
            early_stop_after = 6  # steps
            for epoch in range(config["hyperparameters"]["epochs"]):
                self.epoch = epoch
                self.train(model, lossfunction, optimizer, train_iter, config)
                train_loss, train_acc, _, train_F1 = self.validate(model, lossfunction, train_iter, config,
                                                                   log_results=False)
                sanity_loss, sanity_acc, sanity_acc_per_level, sanity_F1 = self.validate(model, lossfunction,
                                                                                         sanity_iter,
                                                                                         config, log_results=False)
                validation_loss, validation_acc, val_acc_per_level, val_F1 = self.validate(model, lossfunction,
                                                                                           dev_iter,
                                                                                           config, log_results=False)
                saved = False
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    best_val_los_epoch = epoch
                    best_loss_F1 = val_F1
                    if val_F1 > self.save_treshold:
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

                    if val_F1 > self.save_treshold and not saved:
                        # Map to CPU before saving, because this requires additional memory /for some reason/
                        model.to(torch.device("cpu"))
                        torch.save(model,
                                   f"saved/BIG_checkpoint_{str(self.__class__)}_F1"
                                   f"_{val_F1:.5f}_L_{validation_loss}_{get_timestamp()}_{socket.gethostname()}.pt")
                        model.to(device)

                logging.info(
                    f"Epoch {epoch}, Training loss|acc|F1: {train_loss:.6f}|{train_acc:.6f}|{train_F1:.6f}")
                logging.info(
                    f"Epoch {epoch}, Sanity loss|acc|F1: {sanity_loss:.6f}|{sanity_acc:.6f}|{sanity_F1:.6f}")
                logging.info(
                    f"Epoch {epoch}, Validation loss|acc|F1: {validation_loss:.6f}|{validation_acc:.6f}|{val_F1:.6f} - "
                    f"(Best {best_val_loss:.4f}|{best_val_acc:4f}|{best_val_F1})")

                if validation_loss > best_val_loss and epoch > best_val_los_epoch + early_stop_after:
                    logging.info("Early stopping...")
                    break
        except KeyboardInterrupt:
            logging.info('-' * 120)
            logging.info('Exit from training early.')
        finally:
            logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')
        return {
            "best_loss": best_val_loss,
            "best_F1": best_val_F1,
            "bestF1_loss": best_F1_loss,
            "bestloss_F1": best_loss_F1
        }

    def validate(self, model: torch.nn.Module, lossfunction: _Loss, dev_iter: Iterator, config: dict, verbose=False,
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
            model.fit()
        return loss, acc, total_acc_per_level, F1

    def finalize_results_logging(self, csvf, loss, f1):
        csvf.close()
        os.rename(self.TMP_FNAME, f"introspection/introspection"
        f"_{str(self.__class__)}_A{f1:.6f}_L{loss:.6f}_{socket.gethostname()}.tsv", )

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
        self.TMP_FNAME = f"introspection/TMP_introspection_{str(self.__class__)}_{socket.gethostname()}.tsv"
        csvf = open(self.TMP_FNAME, mode="w")
        writer = csv.writer(csvf, delimiter='\t')
        writer.writerow(self.__class__.RESULT_HEADER)
        return csvf, writer
