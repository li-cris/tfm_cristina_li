import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def gaussian_kernel(
        self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None
    ):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        l2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(l2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [
            torch.exp(-l2_distance / bandwidth_temp)
            for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
        )
        XX = kernels[:batch_size, :batch_size]  # noqa: N806
        YY = kernels[batch_size:, batch_size:]  # noqa: N806
        XY = kernels[:batch_size, batch_size:]  # noqa: N806
        loss = torch.mean(XX) + torch.mean(YY) - 2*torch.mean(XY)
        return loss



def compute_kld(source, target):
    epsilon=1e-10
    source = torch.mean(source, dim=0, keepdim = True) + epsilon
    target = torch.mean(target, dim=0, keepdim = True) + epsilon
    source_sm = torch.softmax(source, dim=1)
    target_sm = torch.softmax(target, dim=1)
    kld = torch.sum(source_sm * torch.log(source_sm / target_sm))
    return(kld)