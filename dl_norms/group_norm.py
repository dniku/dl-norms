import numpy as np
import torch
import tqdm
from torch import nn

from .affine import AffineChannelwise
from .utils import allclose_or_none


class MyGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        assert num_channels % num_groups == 0
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        if affine:
            self.affine = AffineChannelwise(num_channels)
        else:
            self.affine = None

    def forward(self, x):
        assert len(x.shape) == 4
        b, c, h, w = x.shape
        assert c == self.num_channels  # not really needed unless we use affine
        g = c // self.num_groups

        # All dims except B; in addition, C gets special treatment.
        x = x.reshape(b, self.num_groups, g, h, w)
        mu = x.mean(dim=(2, 3, 4), keepdim=True)
        sigma = x.var(dim=(2, 3, 4), unbiased=False, keepdim=True)

        result = (x - mu) / torch.sqrt(sigma + self.eps)
        result = result.reshape(b, c, h, w)

        if self.affine is not None:
            result = self.affine(result)

        return result


def test_GroupNorm():
    np.random.seed(42)
    torch.manual_seed(42)

    for affine in [True, False]:
        for _ in tqdm.trange(10, desc=f'GroupNorm({affine=})'):
            # layers initialization
            g = np.random.randint(1, 5)
            num_groups = np.random.randint(1, 5)
            n_in = num_groups * g
            input_shape = (8, n_in, 20, 16)
            torch_layer = nn.GroupNorm(num_groups, n_in, affine=affine)
            custom_layer = MyGroupNorm(num_groups, n_in, affine=affine)

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
