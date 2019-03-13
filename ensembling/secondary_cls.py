import torch
import torch.nn.functional as F


class SecondaryCls(torch.nn.Module):
    def __init__(self, config, classes=4):
        super().__init__()
        self.I = torch.eye(4).cuda()
        self.a = torch.nn.Parameter(torch.randn((config["hyperparameters"]["ensemble_models"],)))
        # self.b = torch.nn.Parameter(torch.randn((config["hyperparameters"]["ensemble_models"],)))
        # self.l = torch.nn.Linear(config["hyperparameters"]["ensemble_models"], classes)

    def forward(self, preds):
        a = F.softmax(self.a)
        m = tuple([a[i] * self.I for i in range(self.a.shape[0])])
        l = torch.cat(m, -1)
        return torch.matmul(preds, l.t())