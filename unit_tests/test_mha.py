import sys
import os

import numpy as np
import cupy as cp
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

sys.path.append(".")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from typing import Dict, Tuple

from model.multi_head_attention import (
    MultiHeadAttention,
)


def access_grad(model: nn.Module) -> Tuple[Dict, Dict]:
    dz = [x.grad for x in model.parameters()]
    dz_named = [x for x in model.named_parameters()]
    mapped_grad = {
        dz_named_item[0]: dz_item for dz_named_item, dz_item in zip(dz_named, dz)
    }
    mapped_params = dict(model.named_parameters())
    return mapped_grad, mapped_params


class MHA(nn.Module):
    """Multi head attention block"""

    # ref MHA implementation - https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
    def __init__(self, d, n_heads=2):
        """Initialize.

        Args:
            d: dimension
            n_heads: Number of heads. Defaults to 2.
        """
        super(MHA, self).__init__()
        self.d = d
        self.n_heads = n_heads
        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.k_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.v_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        """Forward propagation.

        Args:
            images: images.

        Returns:
            output of forward propagation.
        """
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head : (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)
                pre_attention = q @ k.T / (self.d_head**0.5)
                attention = self.softmax(pre_attention)
                mul = attention @ v
                seq_result.append(mul)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class TestModel(nn.Module):
    """Sample model under test"""

    def __init__(self):
        """Initialize."""
        super(TestModel, self).__init__()
        self.mlp_1 = nn.Linear(12, 8)
        self.mha = MHA(8)
        self.mlp_2 = nn.Linear(8, 10)

    def mha_grad_out_hook(self, grad):
        """Hook to tap grad out.

        Args:
            grad: grad.
        """
        self.grad_out = grad

    def mha_grad_in_hook(self, grad):
        """Hook to tap grad in.

        Args:
            grad: grad.
        """
        self.grad_in = grad

    def forward(self, images):
        """Forward propagation.

        Args:
            images: images.

        Returns:
            output of forward propagation.
        """
        out1 = self.mlp_1(images)
        out1.register_hook(self.mha_grad_out_hook)
        out2 = self.mha(out1)
        out2.register_hook(self.mha_grad_in_hook)
        out3 = self.mlp_2(out2)
        return out1, out2, out3


def set_parameters_externally_mha(
    mha_object: MultiHeadAttention, mapped_weights: Dict
) -> None:
    """Set weights externally. This is used only for testing.

    Args:
        mha_object: instance of mha.
        mapped_weights: weights from pytorch.
    """
    count = 0
    for q in mha_object.q_mappings:
        namew = f"mha.q_mappings.{count}.weight"
        nameb = f"mha.q_mappings.{count}.bias"
        # Convert NumPy arrays to CuPy arrays
        weight = cp.asarray(mapped_weights[namew].detach().numpy().T)
        bias = cp.asarray(mapped_weights[nameb].detach().numpy())
        q.init_params(weight, bias)
        count += 1
    count = 0
    for k in mha_object.k_mappings:
        namew = f"mha.k_mappings.{count}.weight"
        nameb = f"mha.k_mappings.{count}.bias"
        # Convert NumPy arrays to CuPy arrays
        weight = cp.asarray(mapped_weights[namew].detach().numpy().T)
        bias = cp.asarray(mapped_weights[nameb].detach().numpy())
        k.init_params(weight, bias)
        count += 1
    count = 0
    for v in mha_object.v_mappings:
        namew = f"mha.v_mappings.{count}.weight"
        nameb = f"mha.v_mappings.{count}.bias"
        # Convert NumPy arrays to CuPy arrays
        weight = cp.asarray(mapped_weights[namew].detach().numpy().T)
        bias = cp.asarray(mapped_weights[nameb].detach().numpy())
        v.init_params(weight, bias)
        count += 1


class TestLinearLayer(unittest.TestCase):
    def test_linear(self):
        """Test functioning of linear layer."""
        model = TestModel()
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = CrossEntropyLoss()
        # create dummy inputs
        x = torch.tensor(
            np.random.rand(2, 50, 12), dtype=torch.float, requires_grad=True
        )
        y_ = np.zeros((2, 10))
        y_[0][5] = 1
        y_[1][2] = 1
        y = torch.tensor(y_, dtype=torch.long)

        out1, out2, y_hat = model(x)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        _, mapped_params = access_grad(model)

        # assign
        input = cp.asarray(out1.detach().numpy())
        output = cp.asarray(out2.detach().numpy())
        grad_in = cp.asarray(model.grad_in.detach().numpy())
        grad_out = cp.asarray(model.grad_out.detach().numpy())
        # call our layer
        custom_mha = MultiHeadAttention(8)
        set_parameters_externally_mha(custom_mha, mapped_params)
        custom_out = custom_mha.forward(input)
        # validate
        decimal_place = 2
        message = "NumPy and reference implementation not almost equal."

        np.testing.assert_array_almost_equal(
            cp.asnumpy(custom_out), cp.asnumpy(output), decimal_place, message
        )

        custom_grad_out = custom_mha.backward(grad_in)
        np.testing.assert_array_almost_equal(
            cp.asnumpy(custom_grad_out), cp.asnumpy(grad_out), decimal_place, message
        )


if __name__ == "__main__":
    unittest.main()
