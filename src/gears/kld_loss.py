# kld_loss.py
import torch

def compute_kld(source, target):
    epsilon=1e-10
    source = torch.mean(source, dim=0, keepdim = True) + epsilon
    target = torch.mean(target, dim=0, keepdim = True) + epsilon
    source_sm = torch.softmax(source, dim=1)
    target_sm = torch.softmax(target, dim=1)
    kld = torch.sum(source_sm * torch.log(source_sm / target_sm))
    return(kld)