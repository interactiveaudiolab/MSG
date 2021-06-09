import torch.nn as nn

from functools import reduce


class MLP(nn.Module):
    def __init__(self, layers, layer_activation=nn.ReLU(), out_activation=None):
        super(MLP, self).__init__()

        self.layer_activation = layer_activation
        self.output_activation = out_activation
        self.layers = nn.ModuleList(
            [nn.Linear(inn, out) for inn, out in zip(layers[:-1], layers[1:])]
        )

    def forward(self, x):
        x = reduce(lambda x, f: f(self.layer_activation(x)), self.layers[1:], self.layers[0](x))

        if self.output_activation is not None:
            return self.output_activation(x)
        else:
            return x
