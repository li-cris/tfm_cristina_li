import math

import torch
import torch.nn as nn


class NetworkActivity_layer(torch.nn.Module):
    def __init__(
        self,
        input_genes,
        output_gs,
        relation_dict,
        bias=True,
        device=None,
        dtype=None,
        lambda_parameter=0,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_genes = input_genes
        self.output_gs = output_gs
        self.relation_dict = relation_dict

        ## create sparse weight matrix according to GO relationships
        mask = torch.zeros((self.input_genes, self.output_gs), **factory_kwargs)

        ## set to 1 remaining values
        for i in range(self.input_genes):
            for latent_go in self.relation_dict[i]:
                mask[i, latent_go] = 1

        # include λ
        self.mask = mask
        self.mask[self.mask == 0] = lambda_parameter

        # apply sp
        self.weight = nn.Parameter(
            torch.empty((self.output_gs, self.input_genes), **factory_kwargs)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_gs, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def forward(self, x):
        output = x @ ((self.weight * self.mask.T).T)
        if self.bias is not None:
            return output + self.bias
        return output

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
