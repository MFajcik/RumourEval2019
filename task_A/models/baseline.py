import torch
import torch.nn.functional as F


class Baseline(torch.nn.Module):
    def __init__(self, config, classes=4, vocab=None):
        super().__init__()
        self.encoder = None
        self.fclayers = torch.nn.ModuleList()
        fc_size = config["hyperparameters"]["FC_size"]
        if config["hyperparameters"]["FC_layers"] > 0:
            self.fclayers.append(torch.nn.Linear(config["hyperparameters"]["inp_size"], fc_size))
        for i in range(config["hyperparameters"]["FC_layers"] - 1):
            self.fclayers.append(torch.nn.Linear(fc_size, fc_size))

        self.dropout_rate = config["hyperparameters"]["dropout_rate"]
        self.final_layer = torch.nn.Linear(self.fclayers[-1].weight.shape[0], classes)

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
        h = torch.cat(tuple(used_features), dim=-1).contiguous()
        for fc in self.fclayers:
            h = F.dropout(F.relu(fc(h)), self.dropout_rate)
        return self.final_layer(h)  # bsz x seq_len x classes
