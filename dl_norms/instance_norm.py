import numpy as np
import torch
import tqdm
from torch import nn

from .affine import AffineChannelwise
from .utils import allclose_or_none


class MyInstanceNorm2d(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5, affine=False, track_running_stats=False):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        if affine:
            self.affine = AffineChannelwise(num_features)
        else:
            self.affine = None

        self.track_running_stats = track_running_stats
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        assert len(x.shape) == 4
        b, c, h, w = x.shape

        if self.training or not self.track_running_stats:
            # All dims except for B and C
            mu = x.mean(dim=(2, 3))
            sigma = x.var(dim=(2, 3), unbiased=False)
        else:
            mu, sigma = self.running_mean, self.running_var
            b = 1

        if self.training and self.track_running_stats:
            sigma_unbiased = sigma * ((h * w) / ((h * w) - 1))
            self.running_mean = self.running_mean * (1 - self.momentum) + mu.mean(dim=0) * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + sigma_unbiased.mean(dim=0) * self.momentum

        mu = mu.reshape(b, c, 1, 1)
        sigma = sigma.reshape(b, c, 1, 1)
        result = (x - mu) / torch.sqrt(sigma + self.eps)

        if self.affine is not None:
            result = self.affine(result)

        return result


def test_InstanceNorm2d():
    np.random.seed(42)
    torch.manual_seed(42)

    input_shape = (8, 4, 20, 16)
    n_in = input_shape[1]
    for affine in [False, True]:
        for track_running_stats in [False, True]:
            for i in tqdm.trange(10, desc=f'InstanceNorm({affine=}, {track_running_stats=})'):
                # layers initialization
                momentum = np.random.uniform(0.01, 0.99)  # 0.1
                torch_layer = nn.InstanceNorm2d(n_in, momentum=momentum, affine=affine,
                                                track_running_stats=track_running_stats)
                custom_layer = MyInstanceNorm2d(n_in, momentum=momentum, affine=affine,
                                                track_running_stats=track_running_stats)

                for _ in range(10):
                    torch_layer.train()
                    custom_layer.train()

                    layer_input = np.random.uniform(-5, 5, input_shape).astype(np.float32)
                    torch_layer_input = torch.tensor(layer_input, requires_grad=True)
                    custom_layer_input = torch.tensor(layer_input, requires_grad=True)
                    next_layer_grad = torch.from_numpy(np.random.uniform(-5, 5, input_shape).astype(np.float32))

                    # 1. check layer output
                    torch_layer_output = torch_layer(torch_layer_input)
                    custom_layer_output = custom_layer(custom_layer_input)
                    assert allclose_or_none(torch_layer_output, custom_layer_output, atol=1e-6)

                    # 2. check layer input grad
                    torch_layer_output.backward(next_layer_grad)
                    custom_layer_output.backward(next_layer_grad)
                    assert allclose_or_none(torch_layer_input.grad, custom_layer_input.grad, atol=1e-7)

                    # 3. check running mean & variance
                    assert allclose_or_none(custom_layer.running_mean, torch_layer.running_mean)
                    assert allclose_or_none(custom_layer.running_var, torch_layer.running_var)

                    # 4. check evaluation mode
                    torch_layer.eval()
                    custom_layer.eval()
                    torch_layer_output = torch_layer(torch_layer_input)
                    custom_layer_output = custom_layer(custom_layer_input)
                    assert allclose_or_none(torch_layer_output, custom_layer_output, atol=1e-6)

                    # 5. update parameters so that weight & bias are different in the next step
                    if affine:
                        torch_layer.weight.data.normal_()
                        torch_layer.bias.data.uniform_()

                        custom_layer.affine.weight.data.copy_(torch_layer.weight)
                        custom_layer.affine.bias.data.copy_(torch_layer.bias)
