import torch
import torch.nn.functional as F

from embedders import Embedder
from encoders import SelfAttentiveEncoder


class SelfAttandBsline(torch.nn.Module):
    def __init__(self, config, classes=4, vocab=None):
        super().__init__()
        self.embedder = Embedder(vocab, config["hyperparameters"])
        self.encoder = SelfAttentiveEncoder(config["hyperparameters"])
        self.fclayers = torch.nn.ModuleList()
        fc_size = config["hyperparameters"]["FC_size"]
        if config["hyperparameters"]["FC_layers"] > 0:
            self.fclayers.append(torch.nn.Linear(config["hyperparameters"]["inp_size"], fc_size))
        for i in range(config["hyperparameters"]["FC_layers"] - 1):
            self.fclayers.append(torch.nn.Linear(fc_size, fc_size))

        self.dropout_rate = config["hyperparameters"]["dropout_rate"]
        self.final_layer = torch.nn.Linear(self.fclayers[-1].weight.shape[0] + self.encoder.get_output_dim(), classes)

    def forward(self, batch):
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
        baseline_features = torch.cat(tuple(used_features), dim=-1).contiguous()
        for fc in self.fclayers:
            baseline_features = F.dropout(F.relu(fc(baseline_features)), self.dropout_rate)

        inp = self.prepare_inp(batch)
        flat_inp = inp.view(-1, inp.shape[-1])
        emb = self.embedder(flat_inp)
        text_features, attention = self.encoder(flat_inp, emb, self.embedder.vocab)

        text_features = text_features.view(text_features.shape[0], -1).view(inp.shape[0], inp.shape[1], -1)

        # return self.final_layer(
        #     torch.cat((torch.zeros(text_features.shape).cuda(),
        #                torch.zeros(baseline_features.shape).to(batch.stance_labels.device)), dim=-1))

        # return self.final_layer(
        #     torch.cat((torch.zeros((batch.stance_labels.shape[0], batch.stance_labels.shape[1], 6000)).cuda(),
        #                baseline_features), dim=-1))

        # return self.final_layer(torch.cat((torch.zeros(text_features.shape).to(batch.stance_labels.device), baseline_features), dim=-1))

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
