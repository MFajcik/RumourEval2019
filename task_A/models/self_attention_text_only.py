import torch

from embedders import Embedder
from encoders import SelfAttentiveEncoder


class SelAttTextOnly(torch.nn.Module):
    def __init__(self, config, vocab, classes=4):
        super().__init__()
        self.embedder = Embedder(vocab, config["hyperparameters"])
        self.encoder = SelfAttentiveEncoder(config["hyperparameters"])
        self.dropout_rate = config["hyperparameters"]["dropout_rate"]
        # self.hidden = torch.nn.Linear(self.encoder.get_output_dim(), 314)
        self.final_layer = torch.nn.Linear(self.encoder.get_output_dim(), classes)

    def forward(self, batch):
        inp = self.prepare_inp(batch)
        flat_inp = inp.view(-1, inp.shape[-1])
        emb = self.embedder(flat_inp)
        h, attention = self.encoder(flat_inp, emb, self.embedder.vocab)
        # for fc in self.fclayers:
        #     h = F.dropout(F.relu(fc(h)), self.dropout_rate)

        r = h.view(h.shape[0], -1).view(inp.shape[0], inp.shape[1], -1)
        # r = F.dropout(F.relu(self.hidden(r)), self.dropout_rate)
        return self.final_layer(r), attention

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
