import torch
import torch.nn.functional as F


class SecondaryCls(torch.nn.Module):
    def __init__(self, config, classes=4):
        super().__init__()
        self.I = torch.eye(4).cuda()
        self.a = torch.nn.Parameter(torch.randn((config["hyperparameters"]["ensemble_models"],)))
        # self.b = torch.nn.Parameter(torch.randn((config["hyperparameters"]["ensemble_models"],)))
        # self.l = torch.nn.Linear(config["hyperparameters"]["ensemble_models"], classes)

    # 1  2  3
    # 1  2
    #
    #
    def forward(self, preds):
        a = F.softmax(self.a)
        m = tuple([a[i] * self.I for i in range(self.a.shape[0])])
        l = torch.cat(m, -1)
        return torch.matmul(preds, l.t())

# class SecondaryCls(torch.nn.Module):
#     def __init__(self, config, classes=4):
#         super().__init__()
#         self.I = torch.eye(4).cuda()
#         #self.a = torch.nn.Parameter(torch.randn((config["hyperparameters"]["ensemble_models"],)))
#         self.a = torch.Tensor([4.7952e-02, 2.1576e-03, 5.1597e-03, 1.3440e-03, 5.0038e-02, 2.3493e-04,
#         4.6728e-04, 1.1529e-03, 1.7907e-03, 5.4993e-04, 5.0124e-03, 1.5886e-02,
#         5.3100e-02, 5.1245e-03, 9.1452e-03, 1.2458e-03, 9.5360e-04, 8.5072e-03,
#         3.8777e-04, 5.0680e-02, 2.3975e-03, 1.6138e-04, 5.3788e-03, 2.6079e-03,
#         2.2261e-03, 5.7379e-02, 2.8377e-04, 9.6164e-04, 2.8531e-03, 2.7594e-03,
#         1.2829e-03, 4.6040e-03, 8.6004e-03, 2.8204e-04, 5.3947e-03, 6.6437e-03,
#         1.5323e-01, 2.1794e-03, 2.5761e-03, 1.0269e-02, 7.0771e-04, 7.2292e-03,
#         1.1057e-03, 2.4858e-03, 2.0384e-03, 4.5144e-02, 1.9698e-02, 1.7627e-03,
#         1.6253e-03, 3.5965e-04, 1.0519e-03, 1.0929e-02, 1.3086e-03, 2.0433e-04,
#         1.9587e-03, 3.2428e-03, 1.3521e-03, 6.9652e-04, 1.4944e-03, 1.0168e-03,
#         7.3790e-03, 1.5602e-03, 6.5804e-04, 6.2742e-04, 8.2076e-04, 4.9575e-04,
#         1.3101e-03, 1.8814e-03, 2.1895e-02, 1.5332e-03, 3.7174e-03, 3.0868e-01,
#         3.7274e-03, 4.3851e-04, 7.2686e-04, 6.1752e-03]).cuda()
#         #self.b = torch.nn.Parameter(torch.randn((config["hyperparameters"]["ensemble_models"],)))
#         #self.l = torch.nn.Linear(config["hyperparameters"]["ensemble_models"], classes)
#
#
#     #1  2  3
#     # 1  2
#     #
#     #
#     def forward(self, preds):
#         a =self.a #= F.softmax(self.a)
#         m = tuple([a[i] * self.I for i in range (self.a.shape[0])])
#         l = torch.cat(m, -1)
#         return torch.matmul(preds,l.t())
