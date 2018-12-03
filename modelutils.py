import torch


def glorot_param_init(model):
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
