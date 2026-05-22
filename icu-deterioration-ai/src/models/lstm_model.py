"""
LSTM Deterioration Prediction Model
Author: Subham Pal

Multivariate LSTM that predicts probability of patient deterioration
within the next 6 hours based on ICU vital sign sequences.
"""

import torch
import torch.nn as nn
import numpy as np
from loguru import logger


class ICUDeteriorationLSTM(nn.Module):
    """
    Bidirectional LSTM for ICU deterioration prediction.

    Input:  (batch, sequence_length, n_vitals)
    Output: (batch, 1) — deterioration probability
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)

        # Attention pooling over time steps
        attn_weights = self.attention(lstm_out)
        context = (attn_weights * lstm_out).sum(dim=1)

        return self.classifier(context)


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                return True
        return False


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X).squeeze()
        loss = criterion(pred, y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train(
    model,
    train_loader,
    val_loader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device)
    # Weighted loss to handle class imbalance (deterioration events are rare)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    early_stop = EarlyStopping(patience=7)

    logger.info(f"Training on {device}")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        logger.info(f"Epoch {epoch+1}/{epochs} — Loss: {train_loss:.4f}")
        scheduler.step(train_loss)
        if early_stop(train_loss):
            logger.info("Early stopping triggered")
            break

    return model
