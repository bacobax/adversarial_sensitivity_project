import torch
import torch.nn as nn
class ChannelLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, pool = None) -> None:
        super(ChannelLinear, self).__init__(in_features, out_features, bias)
        self.compute_axis = 1
        self.pool = pool

    def forward(self, x):
        axis_ref = len(x.shape)-1
        x = torch.transpose(x, self.compute_axis, axis_ref)
        out_shape = list(x.shape); out_shape[-1] = self.out_features
        x = x.reshape(-1, x.shape[-1])
        x = x.matmul(self.weight.t())
        if self.bias is not None:
            x = x + self.bias[None,:]
        x = torch.transpose(x.view(out_shape), axis_ref, self.compute_axis)
        if self.pool is not None:
            x = self.pool(x)
        return x

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output