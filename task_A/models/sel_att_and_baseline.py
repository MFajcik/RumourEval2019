import torch
import torch.nn.functional as F

from neural_bag.embedders import Embedder
from neural_bag.encoders import SelfAttentiveEncoder


class SelfAttandBsline(torch.nn.Module):
    def __init__(self, config, classes=4, vocab=None):
        super().__init__()
        self.baseline_feature_extractor = BaselineFeatureExtractor(config)
        self.textselfatt_feature_extractor = TextualFeatureExtractor(config, vocab)

        # self.final_layer_handf = torch.nn.Linear(self.baseline_feature_extractor.fclayers[-1].weight.shape[0], classes)
        # self.final_layer_text = torch.nn.Linear(self.textselfatt_feature_extractor.encoder.get_output_dim(), classes)
        self.final_layer = torch.nn.Linear(self.baseline_feature_extractor.fclayers[-1].weight.shape[
                                               0] + self.textselfatt_feature_extractor.encoder.get_output_dim(),
                                           classes)

    def get_encoder(self):
        return self.textselfatt_feature_extractor.encoder

    def forward(self, batch, selected_features):
        #
        # return self.final_layer(
        #     torch.cat((text_features,
        #                torch.zeros(baseline_features.shape).to(batch.stance_labels.device)), dim=-1))

        # 2
        # return self.final_layer(
        #     torch.cat((torch.zeros((batch.stance_labels.shape[0], batch.stance_labels.shape[1], 6000)).to(
        #         batch.stance_labels.device),
        #                baseline_features), dim=-1))
        if selected_features == "text":
            text_features = self.textselfatt_feature_extractor(batch)
            baseline_features = self.baseline_feature_extractor(batch)
            features = torch.cat((
                text_features,
                torch.zeros(baseline_features.shape).to(batch.stance_labels.device)),
                dim=-1)
            final_layer = self.final_layer
        elif selected_features == "hand":
            baseline_features = self.baseline_feature_extractor(batch)
            features = torch.cat((
                torch.zeros((batch.stance_labels.shape[0], batch.stance_labels.shape[1], 6000)).to(
                    batch.stance_labels.device),
                baseline_features), dim=-1)
            final_layer = self.final_layer
        else:
            text_features = self.textselfatt_feature_extractor(batch)
            baseline_features = self.baseline_feature_extractor(batch)
            features = torch.cat((text_features, baseline_features), dim=-1)
            # alpha = F.sigmoid(self.attvectorB(F.tanh(self.attmatrixA(features))))
            #
            # features = torch.cat((alpha * text_features, (1 - alpha) * baseline_features), dim=-1)
            final_layer = self.final_layer
        return final_layer(features)


class SelfAttandBslineX(torch.nn.Module):
    def __init__(self, config, classes=4, vocab=None):
        super().__init__()
        self.baseline_feature_extractor = BaselineFeatureExtractor(config)
        self.textselfatt_feature_extractor = TextualFeatureExtractor(config, vocab)

        self.hidden_t = torch.nn.Linear(self.textselfatt_feature_extractor.encoder.get_output_dim(),
                                        self.baseline_feature_extractor.fclayers[-1].weight.shape[0])

        self.dropout_rate = config["hyperparameters"]["dropout_rate"]

        self.final_layer_handf = torch.nn.Linear(self.baseline_feature_extractor.fclayers[-1].weight.shape[0], classes)
        self.final_layer_text = torch.nn.Linear(self.baseline_feature_extractor.fclayers[-1].weight.shape[0], classes)
        self.final_layer = torch.nn.Linear(self.baseline_feature_extractor.fclayers[-1].weight.shape[
                                               0] * 2,
                                           classes)

    def get_encoder(self):
        return self.textselfatt_feature_extractor.encoder

    def forward(self, batch, selected_features):
        #
        # return self.final_layer(
        #     torch.cat((text_features,
        #                torch.zeros(baseline_features.shape).to(batch.stance_labels.device)), dim=-1))

        # 2
        # return self.final_layer(
        #     torch.cat((torch.zeros((batch.stance_labels.shape[0], batch.stance_labels.shape[1], 6000)).to(
        #         batch.stance_labels.device),
        #                baseline_features), dim=-1))
        if selected_features == "text":
            features = F.relu(self.hidden_t(self.textselfatt_feature_extractor(batch)), self.dropout_rate)
            final_layer = self.final_layer_text
        elif selected_features == "hand":
            features = self.baseline_feature_extractor(batch)
            final_layer = self.final_layer_handf
        else:
            text_features = F.relu(self.hidden_t(self.textselfatt_feature_extractor(batch)), self.dropout_rate)
            baseline_features = self.baseline_feature_extractor(batch)
            features = torch.cat((text_features, baseline_features), dim=-1)
            final_layer = self.final_layer
        return final_layer(features)


class BaselineFeatureExtractor(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fclayers = torch.nn.ModuleList()
        fc_size = config["hyperparameters"]["FC_size"]
        if config["hyperparameters"]["FC_layers"] > 0:
            self.fclayers.append(torch.nn.Linear(config["hyperparameters"]["inp_size"], fc_size))
        for i in range(config["hyperparameters"]["FC_layers"] - 1):
            self.fclayers.append(torch.nn.Linear(fc_size, fc_size))

        self.dropout_rate = config["hyperparameters"]["dropout_rate"]

    def forward(self, batch):
        baseline_features = self.prepare_inp(batch)
        for fc in self.fclayers:
            baseline_features = F.dropout(F.relu(fc(baseline_features)), self.dropout_rate)

        return baseline_features

    def prepare_inp(self, batch):
        used_features = [
            batch.hasnegation, batch.hasswearwords, batch.capitalratio,
            batch.hasperiod, batch.hasqmark, batch.hasemark, batch.hasurl, batch.haspic,
            batch.charcount, batch.wordcount, batch.issource,
            batch.Word2VecSimilarityWrtOther,
            batch.Word2VecSimilarityWrtSource,
            batch.Word2VecSimilarityWrtPrev
        ]
        used_features = [batch.avgw2v.cuda()] + \
                        [f.unsqueeze(-1) for f in used_features]
        return torch.cat(tuple(used_features), dim=-1).contiguous()


class TextualFeatureExtractor(torch.nn.Module):

    def __init__(self, config, vocab=None):
        super().__init__()
        self.embedder = Embedder(vocab, config["hyperparameters"])
        self.encoder = SelfAttentiveEncoder(config["hyperparameters"])
        self.dropout_rate = config["hyperparameters"]["dropout_rate"]

    def forward(self, batch):
        inp = self.prepare_inp(batch)
        flat_inp = inp.view(-1, inp.shape[-1])
        emb = self.embedder(flat_inp)
        text_features, attention = self.encoder(flat_inp, emb, self.embedder.vocab)
        return text_features.view(text_features.shape[0], -1).view(inp.shape[0], inp.shape[1], -1)

    def prepare_inp(self, batch):
        maxlen = 0
        for i, textsequence in enumerate(batch.spacy_processed_text_raw):
            for j, text in enumerate(textsequence):
                l = len(text.split())
                if l > maxlen:
                    maxlen = l
        s = batch.stance_labels.shape
        inp = torch.ones((s[0], s[1], maxlen)) * self.embedder.vocab.stoi["<pad>"]
        # Batch x resplen  x MAXSEQ
        for i, textsequence in enumerate(batch.spacy_processed_text_raw):
            for j, text in enumerate(textsequence):
                for k, token in enumerate(text.split()):
                    inp[i, j, k] = self.embedder.vocab.stoi[token]
        return inp.type(torch.long).to(batch.stance_labels.device)
