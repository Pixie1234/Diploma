import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import (
    N_FEATURES,
    OPEN_IDX,
    CLOSE_IDX,
    SEQ_LEN,
    inverse_transform_col,
)


DEVICE = torch.device("cpu")


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


class InformerStyleEncoder(nn.Module):
    """A practical Informer-like encoder for small numeric time series.

    This is not a full ProbSparse Informer implementation, but it follows the
    same idea: attention-based temporal modeling with an encoder stack.
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 64,
        dropout: float = 0.15,
        close_loss_weight: float = 2.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.close_loss_weight = float(close_loss_weight)

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)

        # Next-step prediction from last timestep.
        last = h[:, -1, :]
        return self.head(last)

    def weighted_smooth_l1(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        # y_true/y_pred: (batch, 2) in scaled log-return space.
        # SmoothL1 per element, weighted on Close.
        loss_open = nn.functional.smooth_l1_loss(
            y_pred[:, 0], y_true[:, 0], beta=1.0, reduction="mean"
        )
        loss_close = nn.functional.smooth_l1_loss(
            y_pred[:, 1], y_true[:, 1], beta=1.0, reduction="mean"
        )
        return loss_open + self.close_loss_weight * loss_close


@dataclass
class InformerConfig:
    # Keep the model small so training is fast on CPU.
    d_model: int = 24
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 48
    dropout: float = 0.15
    close_loss_weight: float = 2.0

    lr: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 15
    patience: int = 4


def _make_model(cfg: InformerConfig) -> InformerStyleEncoder:
    return InformerStyleEncoder(
        n_features=N_FEATURES,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        close_loss_weight=cfg.close_loss_weight,
    ).to(DEVICE)


def predict_informer(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    dl = DataLoader(torch.from_numpy(X).float(), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(DEVICE)
            yb = model(xb)
            preds.append(yb.cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_informer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_path: str,
    cfg: InformerConfig,
) -> nn.Module:
    os.makedirs(os.path.dirname(model_path) if model_path and os.path.dirname(model_path) else ".", exist_ok=True)

    model = _make_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    ds_train = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    ds_val = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None
    patience_left = cfg.patience

    for epoch in range(cfg.max_epochs):
        model.train()
        for xb, yb in dl_train:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = model.weighted_smooth_l1(yb, pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = model(xb)
                val_losses.append(float(model.weighted_smooth_l1(yb, pred).cpu().item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, model_path)
    return model


def load_or_train_informer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_path: str,
    cfg: Optional[InformerConfig] = None,
):
    cfg = cfg or InformerConfig()

    if model_path and os.path.exists(model_path):
        try:
            blob = torch.load(model_path, map_location=DEVICE)
            model = _make_model(cfg)
            model.load_state_dict(blob["state_dict"], strict=True)
            model.eval()
            return model, True
        except Exception:
            # Fall back to training.
            pass

    model = train_informer(X_train, y_train, X_val, y_val, model_path, cfg)
    return model, False


def forecast_ohlcv_informer(
    model: nn.Module,
    last_sequence: np.ndarray,
    days: int,
    scaler,
    raw_ohlcv: np.ndarray,
    batch_eval: int = 256,
) -> Dict[str, list]:
    """Autoregressive multi-step forecast for Open/Close (mirrors LSTM logic)."""
    model.eval()

    predictions = {
        "open_prices": [],
        "close_prices": [],
        "open_returns": [],
        "close_returns": [],
    }

    seq = last_sequence.copy()
    base_prices = raw_ohlcv[-1]
    current_price = {"Open": float(base_prices[0]), "Close": float(base_prices[3])}

    for _ in range(days):
        x_in = torch.from_numpy(seq.reshape(1, SEQ_LEN, N_FEATURES)).float().to(DEVICE)
        with torch.no_grad():
            pred_scaled = model(x_in).cpu().numpy()[0]

        open_ret_scaled = float(pred_scaled[0])
        close_ret_scaled = float(pred_scaled[1])

        open_ret = float(inverse_transform_col(open_ret_scaled, OPEN_IDX, scaler))
        close_ret = float(inverse_transform_col(close_ret_scaled, CLOSE_IDX, scaler))

        current_price["Open"] *= float(np.exp(open_ret))
        current_price["Close"] *= float(np.exp(close_ret))

        predictions["open_returns"].append(open_ret)
        predictions["close_returns"].append(close_ret)
        predictions["open_prices"].append(round(current_price["Open"], 2))
        predictions["close_prices"].append(round(current_price["Close"], 2))

        new_seq = np.roll(seq, -1, axis=0)
        next_feat = new_seq[-1].copy()

        next_feat[OPEN_IDX] = open_ret_scaled
        next_feat[CLOSE_IDX] = close_ret_scaled

        # Keep the same heuristic roll-forward as LSTM baseline.
        next_feat[1] = close_ret_scaled  # High
        next_feat[2] = close_ret_scaled  # Low
        next_feat[4] = close_ret_scaled  # Volume

        new_seq[-1] = next_feat
        seq = new_seq

    return predictions
