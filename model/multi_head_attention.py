import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple
from abc import ABC, abstractmethod
import cupy as cp
import copy

from model.linear import Linear
from model.softmax import Softmax


class MultiHeadAttention:
    """Multi head attention"""

    def __init__(self, dimension: int, n_heads: int = 2) -> None:
        """Initialize.

        Args:
            dimension: input dimension.
            n_heads: number of heads.
        """
        self.n_heads = n_heads
        self.d_head = int(dimension / n_heads)
        # Q K V has a d_head * d_head size
        self.q_mappings = [
            Linear(self.d_head, self.d_head) for _ in range(self.n_heads)
        ]
        self.k_mappings = [
            Linear(self.d_head, self.d_head) for _ in range(self.n_heads)
        ]
        self.v_mappings = [
            Linear(self.d_head, self.d_head) for _ in range(self.n_heads)
        ]
        self.softmax = [Softmax() for _ in range(self.n_heads)]

    def forward(self, sequences: cp.ndarray) -> cp.ndarray:
        """Forward propagation with parallel computation for heads.

        Args:
            sequences: input array.

        Returns:
            computed multi-head attention layer output.
        """
        self.sequences = sequences
        self.scale = cp.sqrt(self.d_head)

        # Split sequences into heads and stack for parallelism
        sequences = cp.stack(
            cp.split(sequences, self.n_heads, axis=-1), axis=0
        )  # (n_heads, batch, seq_len, d_head)

        # Apply linear mappings for Q, K, V in parallel
        q_seqs = cp.stack(
            [q_mapping(seq) for q_mapping, seq in zip(self.q_mappings, sequences)],
            axis=0,
        )
        k_seqs = cp.stack(
            [k_mapping(seq) for k_mapping, seq in zip(self.k_mappings, sequences)],
            axis=0,
        )
        v_seqs = cp.stack(
            [v_mapping(seq) for v_mapping, seq in zip(self.v_mappings, sequences)],
            axis=0,
        )

        # Compute scaled dot-product attention for all heads
        scores = (
            cp.matmul(q_seqs, k_seqs.transpose(0, 1, 3, 2)) / self.scale
        )  # (n_heads, batch, seq_len, seq_len)
        attention = cp.stack(
            [softmax(score) for softmax, score in zip(self.softmax, scores)], axis=0
        )

        # Compute attention-weighted values
        result = cp.matmul(attention, v_seqs)  # (n_heads, batch, seq_len, d_head)

        # Combine results across heads and return
        self.result = cp.concatenate(
            cp.split(result, self.n_heads, axis=0), axis=-1
        ).squeeze(
            0
        )  # (batch, seq_len, dimension)
        self.q_seqs, self.k_seqs, self.v_seqs, self.attention_seqs = (
            q_seqs,
            k_seqs,
            v_seqs,
            attention,
        )
        return self.result

    # def forward(self, sequences: cp.ndarray) -> cp.ndarray:
    #     """Forward propagation.

    #     Args:
    #         sequences: input array.

    #     Returns:
    #         computed multi head attention layer output.
    #     """
    #     self.sequences = sequences
    #     self.scale = cp.sqrt(self.d_head)
    #     # convert to list of n_heads elements with info of size (N, seq_length, dimension / n_heads)
    #     sequences = cp.split(sequences, self.n_heads, axis=-1)
    #     result = []
    #     q_seq = []
    #     k_seq = []
    #     v_seq = []
    #     attention_seq = []
    #     for head in range(self.n_heads):
    #         q_mapping = self.q_mappings[head]
    #         k_mapping = self.k_mappings[head]
    #         v_mapping = self.v_mappings[head]
    #         seq = sequences[head]
    #         q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)
    #         q_seq.append(q)
    #         k_seq.append(k)
    #         v_seq.append(v)
    #         attention_seq_head = self.softmax[head](
    #             q @ k.transpose(0, 2, 1) / self.scale
    #         )
    #         attention_seq.append(attention_seq_head)
    #         result.append(attention_seq_head @ v)
    #     # convert to (N, seq_length, dimension)
    #     self.result = cp.dstack(result)
    #     self.q_seqs = cp.dstack(q_seq)
    #     self.k_seqs = cp.dstack(k_seq)
    #     self.v_seqs = cp.dstack(v_seq)
    #     self.attention_seqs = cp.dstack(attention_seq)
    #     return self.result

    def backward(self, error: cp.ndarray) -> None:
        """Backward propagation..

        Args:
            grad: represents the gradient w.r.t. the output. Defaults to None.

        Returns:
            the gradients w.r.t. the input.
        """
        error_head_split = cp.split(error, self.n_heads, axis=-1)
        attention_seqs_split = cp.split(self.attention_seqs, self.n_heads, axis=-1)
        q_seqs_split = cp.split(self.q_seqs, self.n_heads, axis=-1)
        k_seqs_split = cp.split(self.k_seqs, self.n_heads, axis=-1)
        v_seqs_split = cp.split(self.v_seqs, self.n_heads, axis=-1)

        final_error = []
        for i in range(self.n_heads):
            err_attn = error_head_split[i] @ v_seqs_split[i].transpose(0, 2, 1)
            pre_attn_error = self.softmax[i].backward(err_attn)
            v_grad_in = attention_seqs_split[i].transpose(0, 2, 1) @ error_head_split[i]
            error_v_out_i = self.v_mappings[i].backward(v_grad_in)

            k_error = (q_seqs_split[i].transpose(0, 2, 1) @ pre_attn_error) / (
                self.d_head**0.5
            )
            k_error = k_error.transpose(0, 2, 1)
            error_k_out_i = self.k_mappings[i].backward(k_error)

            q_error = (pre_attn_error @ k_seqs_split[i]) / (self.d_head**0.5)
            error_q_out_i = self.q_mappings[i].backward(q_error)

            seq_error = error_q_out_i + error_k_out_i + error_v_out_i
            final_error.append(seq_error)
        return cp.dstack(final_error)

    def init_optimizer(self, optimizer: object) -> None:
        """Initializes optimizers.

        Args:
            optimizer: optimizer.
        """
        for v_mapping in self.v_mappings:
            v_mapping.init_optimizer(optimizer)
        for q_mapping in self.q_mappings:
            q_mapping.init_optimizer(optimizer)
        for k_mapping in self.k_mappings:
            k_mapping.init_optimizer(optimizer)

    def update_params(self) -> None:
        """Update weights based on the calculated gradients."""
        for v_mapping in self.v_mappings:
            v_mapping.update_params()
        for q_mapping in self.q_mappings:
            q_mapping.update_params()
        for k_mapping in self.k_mappings:
            k_mapping.update_params()

    def __call__(self):
        return self.forward(self.sequences)
