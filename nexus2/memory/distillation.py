"""LSTM -> Conv1D distillation trainer.

Trains a Conv1D encoder to match the key/value outputs of a trained LSTM
encoder, achieving ~28x faster inference with minimal accuracy loss.
"""

import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .encoder import Conv1DEncoder, LSTMEncoder


class DistillationTrainer:
    """Distill encoder knowledge into Conv1D encoder."""

    def __init__(
        self,
        teacher: nn.Module,  # LSTMEncoder or MambaEncoder
        student: Conv1DEncoder,
        embedding: nn.Embedding,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 0.01,
    ):
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.embedding = embedding.to(device).eval()
        self.device = device

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.embedding.parameters():
            p.requires_grad = False

        self.optimizer = AdamW(
            self.student.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5,
        )

    def train_epoch(
        self,
        data_tokens: torch.Tensor,
        batch_size: int = 32,
    ) -> float:
        """Train one epoch on token data.

        Args:
            data_tokens: [N, seq_len] integer token indices
            batch_size: training batch size

        Returns:
            Average loss for the epoch.
        """
        self.student.train()
        n = data_tokens.shape[0]
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            batch_idx = perm[i:i + batch_size]
            tokens = data_tokens[batch_idx].to(self.device)
            embeds = self.embedding(tokens)

            # Teacher targets (no grad)
            with torch.no_grad():
                teacher_keys, teacher_vals = self.teacher(embeds)

            # Student predictions
            student_keys, student_vals = self.student(embeds)

            # MSE loss on both keys and values
            loss_keys = F.mse_loss(student_keys, teacher_keys)
            loss_vals = F.mse_loss(student_vals, teacher_vals)
            loss = loss_keys + loss_vals

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        self.scheduler.step(avg_loss)
        return avg_loss

    def train(
        self,
        data_tokens: torch.Tensor,
        epochs: int = 50,
        batch_size: int = 32,
        target_loss: float = 0.001,
    ) -> float:
        """Full distillation training loop.

        Returns final loss.
        """
        best_loss = float("inf")
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(data_tokens, batch_size)
            if loss < best_loss:
                best_loss = loss

            print(
                f"  [distill {epoch}/{epochs}] loss={loss:.6f} best={best_loss:.6f}",
                flush=True,
            )

            if loss < target_loss:
                print(f"  Distillation converged at epoch {epoch}", flush=True)
                break

        return best_loss
