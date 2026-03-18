from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_codex"))

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset


"""
Multivariate + exogenous residual benchmark.

Source of multivariate/exogenous design
---------------------------------------
- Reuses the lagged feature engineering idea from oil_patchtst_nlinear_lgbm.py
- Keeps leakage-conscious exogenous handling via shift(1)
- Extends the experiment family to:
  PatchTST baseline
  PatchTST + NLinear residual
  PatchTST + XGBoost residual
  PatchTST + LightGBM residual

Evaluation protocol
-------------------
- Same outer report protocol as the univariate 0318 benchmark:
  expanding-window ts-cv + final holdout
- Inside each 12-week evaluation window, the baseline is a one-step model applied
  over the 12 evaluation points using lagged exogenous features only.
"""


TARGET_MAP = {
    "WTI Oil": "Com_CrudeOil",
    "Brent Oil": "Com_BrentCrudeOil",
}

FEATURE_GROUPS = {
    "oil_prices": ["Com_CrudeOil", "Com_BrentCrudeOil"],
    "energy_chain": ["Com_Gasoline"],
    "substitutes": ["Com_NaturalGas", "Com_Uranium", "Com_Coal"],
    "macro_industry": ["Com_LME_Cu_Cash", "Com_Steel", "Com_Iron_Ore"],
    "fx_dollar": ["Idx_DxyUSD", "EX_USD_CNY"],
    "rates_bonds": ["Bonds_US_10Y", "Bonds_US_2Y", "Bonds_US_3M"],
    "risk_safehaven": ["Idx_SnPVIX", "Com_Gold"],
    "demand_proxy": ["Idx_SnP500", "Idx_CSI300"],
    "asia_importers": ["EX_USD_KRW", "Bonds_KOR_10Y", "EX_USD_JPY"],
    "biofuel_commodity": ["Com_Corn", "Com_Soybeans", "Com_PalmOil"],
}


@dataclass(frozen=True)
class ProtocolConfig:
    input_size: int = 48
    window_size: int = 12
    step_size: int = 4
    n_windows: int = 24
    final_holdout: int = 12
    season_length: int = 52


@dataclass(frozen=True)
class PatchTSTConfig:
    hidden_size: int = 128
    attention_heads: int = 16
    linear_hidden_size: int = 256
    patch_len: int = 16
    stride: int = 8
    dropout: float = 0.2
    encoder_layers: int = 3
    attn_dropout: float = 0.0
    fc_dropout: float = 0.2
    max_steps: int = 3000
    learning_rate: float = 1e-4
    batch_size: int = 32
    patience: int = 20
    weight_decay: float = 1e-5


@dataclass(frozen=True)
class NLinearConfig:
    max_steps: int = 1500
    learning_rate: float = 1e-3
    batch_size: int = 64
    patience: int = 20
    weight_decay: float = 0.0


@dataclass(frozen=True)
class TreeConfig:
    n_estimators: int = 400
    learning_rate: float = 0.03
    max_depth: int = 4
    num_leaves: int = 31
    min_child_samples: int = 20
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50


@dataclass(frozen=True)
class EvalWindow:
    split: str
    window_id: int
    train_end: int
    test_start: int
    test_end: int


class ResidualDatasetWithExog(Dataset):
    def __init__(self, resid: np.ndarray, exog: np.ndarray, seq_len: int):
        self.resid = resid.astype(np.float32)
        self.exog = exog.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.resid) - self.seq_len

    def __getitem__(self, idx: int):
        end = idx + self.seq_len
        return (
            torch.tensor(self.resid[idx:end], dtype=torch.float32),
            torch.tensor(self.exog[end - 1], dtype=torch.float32),
            torch.tensor(self.resid[end], dtype=torch.float32),
        )


class PatchTSTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ff_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.attn_dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.ff_dropout(ff_out))
        return x


class PatchTSTOneStep(nn.Module):
    def __init__(self, input_size: int, n_features: int, config: PatchTSTConfig):
        super().__init__()
        n_patches = (input_size - config.patch_len) // config.stride + 1
        if n_patches <= 0:
            raise ValueError("Invalid patch configuration for input_size.")
        self.input_size = input_size
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.n_features = n_features
        self.patch_proj = nn.Linear(config.patch_len * n_features, config.hidden_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.input_dropout = nn.Dropout(config.dropout)
        self.encoder = nn.ModuleList(
            [
                PatchTSTBlock(
                    d_model=config.hidden_size,
                    n_heads=config.attention_heads,
                    d_ff=config.linear_hidden_size,
                    dropout=config.dropout,
                    attn_dropout=config.attn_dropout,
                )
                for _ in range(config.encoder_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(n_patches * config.hidden_size, config.linear_hidden_size),
            nn.GELU(),
            nn.Dropout(config.fc_dropout),
            nn.Linear(config.linear_hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        patches = []
        for start in range(0, self.input_size - self.patch_len + 1, self.stride):
            patch = x[:, start : start + self.patch_len, :].reshape(batch, -1)
            patches.append(patch)
        z = torch.stack(patches, dim=1)
        z = self.patch_proj(z)
        z = self.input_dropout(z + self.pos_embedding)
        for block in self.encoder:
            z = block(z)
        return self.head(z).squeeze(-1)


class NLinearWithExog(nn.Module):
    def __init__(self, seq_len: int, n_exog: int, hidden_size: int = 64):
        super().__init__()
        self.linear_resid = nn.Linear(seq_len, hidden_size)
        self.linear_exog = nn.Linear(n_exog, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(hidden_size * 2, 1)

    def forward(self, x_seq: torch.Tensor, x_exog: torch.Tensor) -> torch.Tensor:
        last = x_seq[:, -1:]
        x_norm = x_seq - last
        h1 = torch.relu(self.linear_resid(x_norm))
        h2 = torch.relu(self.linear_exog(x_exog))
        h = self.dropout(torch.cat([h1, h2], dim=-1))
        return self.out(h) + last


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def nrmse_range(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_range = float(np.max(y_true) - np.min(y_true))
    if y_range <= 1e-12:
        return float("nan")
    return rmse(y_true, y_pred) / y_range


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "NRMSE": nrmse_range(y_true, y_pred),
    }


def slugify(text: str) -> str:
    return text.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def model_label(baseline: str, residual: str) -> str:
    return baseline if residual == "-" else f"{baseline} + {residual}"


def build_feature_frame(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, list[str]]:
    base_features: list[str] = []
    for cols in FEATURE_GROUPS.values():
        for col in cols:
            if col in df.columns and col not in base_features:
                base_features.append(col)

    derived = pd.DataFrame(index=df.index)
    for col in base_features:
        derived[f"{col}_ret"] = df[col].pct_change()

    if {"Bonds_US_10Y", "Bonds_US_2Y"}.issubset(df.columns):
        derived["Spread_US_10Y_2Y"] = df["Bonds_US_10Y"] - df["Bonds_US_2Y"]
    if "Com_Gasoline" in df.columns and target_col in df.columns:
        derived["Spread_Crack"] = df["Com_Gasoline"] - df[target_col]
    if "Com_Gold" in df.columns and target_col in df.columns:
        derived["Ratio_Gold_Oil"] = df["Com_Gold"] / df[target_col]

    for col in ["Com_Gasoline", "Com_NaturalGas", "Idx_SnPVIX", "Idx_DxyUSD"]:
        if col in df.columns:
            ma4 = df[col].rolling(4).mean()
            ma12 = df[col].rolling(12).mean()
            derived[f"{col}_ma4_ratio"] = df[col] / ma4 - 1.0
            derived[f"{col}_ma12_ratio"] = df[col] / ma12 - 1.0

    feature_df = pd.concat([df[base_features], derived], axis=1).shift(1)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_names = list(feature_df.columns)
    return feature_df, feature_names


def make_one_step_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for end in range(seq_len, len(X)):
        xs.append(X[end - seq_len : end])
        ys.append(y[end])
    if not xs:
        raise ValueError("Not enough samples to build one-step sequences.")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def split_timewise(
    x: np.ndarray,
    y: np.ndarray,
    min_val: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(x) < 3:
        raise ValueError("Need at least 3 samples for train/validation split.")
    val_size = min(max(min_val, len(x) // 5), len(x) - 1)
    split_idx = len(x) - val_size
    return (
        x[:split_idx].copy(),
        y[:split_idx].copy(),
        x[split_idx:].copy(),
        y[split_idx:].copy(),
    )


def to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def fit_torch_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    learning_rate: float,
    max_steps: int,
    patience: int,
    weight_decay: float,
    device: torch.device,
) -> tuple[nn.Module, dict[str, float | int]]:
    x_train, y_train, x_val, y_val = split_timewise(x, y, min_val=12)
    train_loader = to_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = to_loader(x_val, y_val, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    steps = 0

    while steps < max_steps and bad_epochs < patience:
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1
            if steps >= max_steps:
                break

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())
        val_loss = float(np.mean(val_losses))
        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

    model.load_state_dict(best_state)
    return model, {
        "best_val_mse": best_val,
        "steps": steps,
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
    }


def predict_patch(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def fit_patchtst_baseline(
    x_train_seq: np.ndarray,
    y_train_norm: np.ndarray,
    config: PatchTSTConfig,
    device: torch.device,
) -> tuple[PatchTSTOneStep, dict[str, float | int]]:
    model = PatchTSTOneStep(
        input_size=x_train_seq.shape[1],
        n_features=x_train_seq.shape[2],
        config=config,
    )
    return fit_torch_model(
        model=model,
        x=x_train_seq,
        y=y_train_norm,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        patience=config.patience,
        weight_decay=config.weight_decay,
        device=device,
    )


def fit_nlinear_residual(
    resid_train: np.ndarray,
    exog_train: np.ndarray,
    seq_len: int,
    config: NLinearConfig,
    device: torch.device,
) -> tuple[NLinearWithExog, dict[str, float | int]]:
    dataset = ResidualDatasetWithExog(resid=resid_train, exog=exog_train, seq_len=seq_len)
    x_seq = []
    x_exog = []
    y = []
    for i in range(len(dataset)):
        a, b, c = dataset[i]
        x_seq.append(a.numpy())
        x_exog.append(b.numpy())
        y.append(c.item())
    x_seq = np.asarray(x_seq, dtype=np.float32)
    x_exog = np.asarray(x_exog, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    x = np.concatenate([x_seq, x_exog], axis=1)

    class WrappedModel(nn.Module):
        def __init__(self, seq_len_inner: int, n_exog: int):
            super().__init__()
            self.seq_len_inner = seq_len_inner
            self.n_exog = n_exog
            self.core = NLinearWithExog(seq_len_inner, n_exog)

        def forward(self, xb: torch.Tensor) -> torch.Tensor:
            x_resid = xb[:, : self.seq_len_inner]
            x_exog_inner = xb[:, self.seq_len_inner : self.seq_len_inner + self.n_exog]
            return self.core(x_resid, x_exog_inner).squeeze(-1)

    model = WrappedModel(seq_len_inner=seq_len, n_exog=exog_train.shape[1])
    wrapped, fit_info = fit_torch_model(
        model=model,
        x=x,
        y=y,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        patience=config.patience,
        weight_decay=config.weight_decay,
        device=device,
    )
    return wrapped.core, fit_info


def predict_nlinear_residual(
    model: NLinearWithExog,
    resid_history: np.ndarray,
    exog_future: np.ndarray,
    seq_len: int,
    device: torch.device,
) -> np.ndarray:
    preds = []
    buffer = list(resid_history.astype(np.float32))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i in range(len(exog_future)):
            x_seq = torch.tensor(np.asarray(buffer[-seq_len:], dtype=np.float32)).unsqueeze(0).to(device)
            x_exog = torch.tensor(exog_future[i : i + 1].astype(np.float32)).to(device)
            pred = model(x_seq, x_exog).detach().cpu().numpy().reshape(-1)[0]
            preds.append(pred)
            buffer.append(pred)
    return np.asarray(preds, dtype=np.float32)


def fit_tree_residual(
    resid_train: np.ndarray,
    exog_train: np.ndarray,
    seq_len: int,
    family: str,
    config: TreeConfig,
    seed: int,
) -> tuple[object, dict[str, int]]:
    x_rows = []
    y_rows = []
    for end in range(seq_len, len(resid_train)):
        feat = np.concatenate([resid_train[end - seq_len : end], exog_train[end - 1]], axis=0)
        x_rows.append(feat)
        y_rows.append(resid_train[end])
    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    x_train, y_train, x_val, y_val = split_timewise(x, y, min_val=12)

    if family == "lgbm":
        model = lgb.LGBMRegressor(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            num_leaves=config.num_leaves,
            max_depth=config.max_depth,
            min_child_samples=config.min_child_samples,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            random_state=seed,
            verbosity=-1,
            n_jobs=-1,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
        )
    elif family == "xgb":
        model = xgb.XGBRegressor(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            min_child_weight=config.min_child_weight,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=config.early_stopping_rounds,
            verbosity=0,
        )
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    else:
        raise ValueError(f"Unsupported family: {family}")

    return model, {
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
    }


def predict_tree_residual(
    model: object,
    resid_history: np.ndarray,
    exog_future: np.ndarray,
    seq_len: int,
) -> np.ndarray:
    preds = []
    buffer = list(resid_history.astype(np.float32))
    for i in range(len(exog_future)):
        feat = np.concatenate([np.asarray(buffer[-seq_len:], dtype=np.float32), exog_future[i]], axis=0)[None, :]
        pred = float(model.predict(feat)[0])
        preds.append(pred)
        buffer.append(pred)
    return np.asarray(preds, dtype=np.float32)


def build_eval_windows(series_length: int, protocol: ProtocolConfig) -> tuple[list[EvalWindow], int]:
    ts_cv_span = protocol.window_size + protocol.step_size * (protocol.n_windows - 1)
    eval_span = ts_cv_span + protocol.final_holdout
    initial_train = series_length - eval_span

    windows = []
    for idx in range(protocol.n_windows):
        train_end = initial_train + idx * protocol.step_size
        test_start = train_end
        test_end = test_start + protocol.window_size
        windows.append(EvalWindow("tscv", idx + 1, train_end, test_start, test_end))

    holdout_start = series_length - protocol.final_holdout
    windows.append(EvalWindow("holdout", 1, holdout_start, holdout_start, series_length))
    return windows, initial_train


def prepare_target_data(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    feature_df, feature_names = build_feature_frame(df, target_col)
    X_all = np.nan_to_num(feature_df.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_all = df[target_col].to_numpy(dtype=np.float32)
    return X_all, y_all, feature_names, df.index


def save_prediction_plots(prediction_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    color_map = {
        "PatchTST": "#7f7f7f",
        "PatchTST + NLinear": "#d62728",
        "PatchTST + XGBoost": "#1f77b4",
        "PatchTST + LightGBM": "#2ca02c",
    }
    plot_rows = []
    for (eval_split, target), group in prediction_df.groupby(["EvalSplit", "Target"], sort=False):
        group = group.copy()
        group["Date"] = pd.to_datetime(group["Date"])
        actual_df = group.groupby("Date", as_index=False)["Actual"].mean().sort_values("Date")

        fig, ax = plt.subplots(figsize=(13, 5.5))
        ax.plot(actual_df["Date"], actual_df["Actual"], color="black", lw=2.6, label="Actual")

        for baseline, residual in [
            ("PatchTST", "-"),
            ("PatchTST", "NLinear"),
            ("PatchTST", "XGBoost"),
            ("PatchTST", "LightGBM"),
        ]:
            sub = group[(group["Baseline"] == baseline) & (group["ResidualModel"] == residual)].copy()
            if sub.empty:
                continue
            pred_df = sub.groupby("Date", as_index=False)["Prediction"].mean().sort_values("Date")
            label = model_label(baseline, residual)
            ax.plot(
                pred_df["Date"],
                pred_df["Prediction"],
                color=color_map.get(label),
                lw=2.0,
                linestyle="--" if residual == "-" else "-",
                label=label,
            )

        ax.set_title(f"{target} | {'TS-CV Mean Prediction vs Actual' if eval_split == 'tscv' else 'Final Holdout Prediction vs Actual'}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, frameon=False)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

        if eval_split == "tscv":
            ax.text(
                0.99,
                0.02,
                "Note: overlapping fold predictions are averaged by date.",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#555555",
            )

        plot_path = plot_dir / f"{eval_split}_{slugify(target)}_actual_vs_pred.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        plot_rows.append(
            {
                "EvalSplit": eval_split,
                "Target": target,
                "PlotPath": str(plot_path),
                "Description": "Actual vs prediction comparison",
            }
        )

    manifest = pd.DataFrame(plot_rows)
    manifest.to_csv(out_dir / "plot_manifest.csv", index=False)
    return manifest


def aggregate_leaderboard(prediction_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["Target", "TargetColumn", "EvalSplit", "Baseline", "ResidualModel"]
    for keys, group in prediction_df.groupby(group_cols, sort=False):
        target, target_col, eval_split, baseline, residual = keys
        metrics = compute_metrics(group["Actual"].to_numpy(dtype=float), group["Prediction"].to_numpy(dtype=float))
        rows.append(
            {
                "Target": target,
                "TargetColumn": target_col,
                "EvalSplit": eval_split,
                "Baseline": baseline,
                "ResidualModel": residual,
                **metrics,
                "NumForecastPoints": int(len(group)),
                "NumWindows": int(group["WindowId"].nunique()),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["EvalSplit", "Target", "RMSE"]).reset_index(drop=True)


def run_one_window(
    target_name: str,
    target_col: str,
    X_all: np.ndarray,
    y_all: np.ndarray,
    dates: pd.DatetimeIndex,
    window: EvalWindow,
    protocol: ProtocolConfig,
    patch_cfg: PatchTSTConfig,
    nlinear_cfg: NLinearConfig,
    tree_cfg: TreeConfig,
    device: torch.device,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    train_end = window.train_end
    test_start = window.test_start
    test_end = window.test_end

    X_train_raw = X_all[:train_end]
    X_test_block_raw = X_all[test_start:test_end]
    y_train = y_all[:train_end]
    y_test = y_all[test_start:test_end]

    scaler_x = RobustScaler()
    X_train_scaled = scaler_x.fit_transform(X_train_raw)
    X_test_scaled = scaler_x.transform(X_test_block_raw)

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std <= 1e-8:
        y_std = 1.0

    y_train_norm = ((y_train - y_mean) / y_std).astype(np.float32)

    Xtr_seq, ytr_seq = make_one_step_sequences(X_train_scaled, y_train_norm, protocol.input_size)
    patch_model, patch_fit = fit_patchtst_baseline(
        x_train_seq=Xtr_seq,
        y_train_norm=ytr_seq,
        config=patch_cfg,
        device=device,
    )

    patch_train_pred_norm = predict_patch(patch_model, Xtr_seq, batch_size=patch_cfg.batch_size, device=device)
    patch_train_pred = patch_train_pred_norm * y_std + y_mean
    y_train_aligned = y_train[protocol.input_size:]
    resid_train = (y_train_aligned - patch_train_pred).astype(np.float32)
    train_exog_aligned = Xtr_seq[:, -1, :].astype(np.float32)

    test_seq_rows = []
    for idx in range(test_start, test_end):
        test_seq_rows.append(X_all[idx - protocol.input_size : idx])
    test_seq_rows = np.asarray(test_seq_rows, dtype=np.float32)
    test_seq_scaled = scaler_x.transform(test_seq_rows.reshape(-1, X_all.shape[1])).reshape(test_seq_rows.shape)
    test_exog_current = test_seq_scaled[:, -1, :].astype(np.float32)

    patch_test_pred_norm = predict_patch(patch_model, test_seq_scaled, batch_size=patch_cfg.batch_size, device=device)
    patch_test_pred = patch_test_pred_norm * y_std + y_mean

    nlinear_model, nlinear_fit = fit_nlinear_residual(
        resid_train=resid_train,
        exog_train=train_exog_aligned,
        seq_len=protocol.input_size,
        config=nlinear_cfg,
        device=device,
    )
    nlinear_resid = predict_nlinear_residual(
        model=nlinear_model,
        resid_history=resid_train[-protocol.input_size :],
        exog_future=test_exog_current,
        seq_len=protocol.input_size,
        device=device,
    )

    xgb_model, xgb_fit = fit_tree_residual(
        resid_train=resid_train,
        exog_train=train_exog_aligned,
        seq_len=protocol.input_size,
        family="xgb",
        config=tree_cfg,
        seed=seed,
    )
    xgb_resid = predict_tree_residual(
        model=xgb_model,
        resid_history=resid_train[-protocol.input_size :],
        exog_future=test_exog_current,
        seq_len=protocol.input_size,
    )

    lgbm_model, lgbm_fit = fit_tree_residual(
        resid_train=resid_train,
        exog_train=train_exog_aligned,
        seq_len=protocol.input_size,
        family="lgbm",
        config=tree_cfg,
        seed=seed,
    )
    lgbm_resid = predict_tree_residual(
        model=lgbm_model,
        resid_history=resid_train[-protocol.input_size :],
        exog_future=test_exog_current,
        seq_len=protocol.input_size,
    )

    predictions = [
        ("PatchTST", "-", patch_test_pred),
        ("PatchTST", "NLinear", patch_test_pred + nlinear_resid),
        ("PatchTST", "XGBoost", patch_test_pred + xgb_resid),
        ("PatchTST", "LightGBM", patch_test_pred + lgbm_resid),
    ]

    metric_rows = []
    pred_rows = []
    test_dates = dates[test_start:test_end]
    for baseline_name, residual_name, pred in predictions:
        metrics = compute_metrics(y_test, pred)
        metric_rows.append(
            {
                "Target": target_name,
                "TargetColumn": target_col,
                "EvalSplit": window.split,
                "WindowId": window.window_id,
                "TrainEndDate": str(dates[train_end - 1].date()),
                "TestStartDate": str(dates[test_start].date()),
                "TestEndDate": str(dates[test_end - 1].date()),
                "TrainSize": int(train_end),
                "TestSize": int(len(y_test)),
                "Baseline": baseline_name,
                "ResidualModel": residual_name,
                **metrics,
            }
        )
        for step_idx, dt in enumerate(test_dates):
            pred_rows.append(
                {
                    "Target": target_name,
                    "TargetColumn": target_col,
                    "EvalSplit": window.split,
                    "WindowId": window.window_id,
                    "Date": str(dt.date()),
                    "HorizonStep": step_idx + 1,
                    "Baseline": baseline_name,
                    "ResidualModel": residual_name,
                    "Actual": float(y_test[step_idx]),
                    "Prediction": float(pred[step_idx]),
                    "Error": float(y_test[step_idx] - pred[step_idx]),
                }
            )

    fit_meta = {
        "Target": target_name,
        "EvalSplit": window.split,
        "WindowId": window.window_id,
        "PatchTST": patch_fit,
        "NLinear": nlinear_fit,
        "XGBoost": xgb_fit,
        "LightGBM": lgbm_fit,
        "FeatureCount": int(X_all.shape[1]),
    }
    return metric_rows, pred_rows, fit_meta


def resolve_targets(requested: list[str] | None) -> dict[str, str]:
    if not requested or requested == ["all"]:
        return TARGET_MAP
    normalized = {k.lower(): k for k in TARGET_MAP}
    selected = {}
    for item in requested:
        key = item.lower()
        if key in normalized:
            label = normalized[key]
            selected[label] = TARGET_MAP[label]
        elif item in TARGET_MAP.values():
            label = next(name for name, col in TARGET_MAP.items() if col == item)
            selected[label] = item
        else:
            raise ValueError(f"Unknown target selection: {item}")
    return selected


def run_experiment(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = ProtocolConfig(
        input_size=args.input_size,
        window_size=args.window_size,
        step_size=args.step_size,
        n_windows=args.n_windows,
        final_holdout=args.final_holdout,
        season_length=args.season_length,
    )
    patch_cfg = PatchTSTConfig(
        max_steps=args.patch_max_steps,
        learning_rate=args.patch_learning_rate,
        batch_size=args.patch_batch_size,
        patience=args.patch_patience,
    )
    nlinear_cfg = NLinearConfig(
        max_steps=args.nlinear_max_steps,
        learning_rate=args.nlinear_learning_rate,
        batch_size=args.nlinear_batch_size,
        patience=args.nlinear_patience,
    )
    tree_cfg = TreeConfig(
        n_estimators=args.tree_n_estimators,
        learning_rate=args.tree_learning_rate,
        early_stopping_rounds=args.tree_early_stopping_rounds,
    )

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    df = pd.read_csv(data_path)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)
    df = df.set_index("dt")

    selected_targets = resolve_targets(args.targets)
    windows, initial_train = build_eval_windows(series_length=len(df), protocol=protocol)

    metric_rows = []
    prediction_rows = []
    fit_meta_rows = []
    feature_meta = {}

    print("=" * 80)
    print("PatchTST Exogenous Residual Benchmark 0318")
    print("=" * 80)
    print(f"Data path        : {data_path}")
    print(f"Observations     : {len(df)}")
    print(f"Date range       : {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"Targets          : {', '.join(selected_targets.keys())}")
    print(f"Device           : {device}")
    print(f"Initial train len: {initial_train}")
    print(f"Initial train end: {df.index[initial_train - 1].date()}")
    print(f"Eval start       : {df.index[initial_train].date()}")
    print(f"Holdout start    : {df.index[len(df) - protocol.final_holdout].date()}")
    print()

    for target_name, target_col in selected_targets.items():
        X_all, y_all, feature_names, dates = prepare_target_data(df, target_col)
        feature_meta[target_name] = {
            "target_col": target_col,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
        }
        print(f"[Target] {target_name} ({target_col}) | features={len(feature_names)}")
        for window in windows:
            print(
                f"  - {window.split} window {window.window_id:02d}: "
                f"train_end={dates[window.train_end - 1].date()}, "
                f"test={dates[window.test_start].date()} ~ {dates[window.test_end - 1].date()}"
            )
            window_metrics, window_predictions, fit_meta = run_one_window(
                target_name=target_name,
                target_col=target_col,
                X_all=X_all,
                y_all=y_all,
                dates=dates,
                window=window,
                protocol=protocol,
                patch_cfg=patch_cfg,
                nlinear_cfg=nlinear_cfg,
                tree_cfg=tree_cfg,
                device=device,
                seed=args.seed,
            )
            metric_rows.extend(window_metrics)
            prediction_rows.extend(window_predictions)
            fit_meta_rows.append(fit_meta)
        print()

    metric_df = pd.DataFrame(metric_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    leaderboard_df = aggregate_leaderboard(prediction_df)
    plot_manifest = save_prediction_plots(prediction_df, out_dir)

    metric_df.to_csv(out_dir / "window_metrics.csv", index=False)
    prediction_df.to_csv(out_dir / "window_predictions.csv", index=False)
    leaderboard_df.to_csv(out_dir / "leaderboard_all.csv", index=False)
    leaderboard_df[leaderboard_df["EvalSplit"] == "tscv"].to_csv(out_dir / "leaderboard_tscv.csv", index=False)
    leaderboard_df[leaderboard_df["EvalSplit"] == "holdout"].to_csv(out_dir / "leaderboard_holdout.csv", index=False)

    metadata = {
        "protocol": asdict(protocol),
        "patchtst": asdict(patch_cfg),
        "nlinear": asdict(nlinear_cfg),
        "tree_model": asdict(tree_cfg),
        "seed": args.seed,
        "device": str(device),
        "target_map": selected_targets,
        "initial_train_len": initial_train,
        "initial_train_end": str(df.index[initial_train - 1].date()),
        "eval_start": str(df.index[initial_train].date()),
        "holdout_start": str(df.index[len(df) - protocol.final_holdout].date()),
        "feature_metadata": feature_meta,
        "fit_metadata": fit_meta_rows,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Saved:")
    for name in [
        "leaderboard_tscv.csv",
        "leaderboard_holdout.csv",
        "window_metrics.csv",
        "window_predictions.csv",
        "plot_manifest.csv",
        "run_metadata.json",
    ]:
        print(f"  - {out_dir / name}")
    for _, row in plot_manifest.iterrows():
        print(f"  - {row['PlotPath']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PatchTST + exogenous residual benchmark for oil targets.")
    parser.add_argument("--data_path", default="data_weekly_260120.csv")
    parser.add_argument("--output_dir", default="output_oil_patchtst_exog_residual_0318")
    parser.add_argument("--targets", nargs="*", default=["all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)

    parser.add_argument("--input_size", type=int, default=48)
    parser.add_argument("--window_size", type=int, default=12)
    parser.add_argument("--step_size", type=int, default=4)
    parser.add_argument("--n_windows", type=int, default=24)
    parser.add_argument("--final_holdout", type=int, default=12)
    parser.add_argument("--season_length", type=int, default=52)

    parser.add_argument("--patch_max_steps", type=int, default=3000)
    parser.add_argument("--patch_learning_rate", type=float, default=1e-4)
    parser.add_argument("--patch_batch_size", type=int, default=32)
    parser.add_argument("--patch_patience", type=int, default=20)

    parser.add_argument("--nlinear_max_steps", type=int, default=1500)
    parser.add_argument("--nlinear_learning_rate", type=float, default=1e-3)
    parser.add_argument("--nlinear_batch_size", type=int, default=64)
    parser.add_argument("--nlinear_patience", type=int, default=20)

    parser.add_argument("--tree_n_estimators", type=int, default=400)
    parser.add_argument("--tree_learning_rate", type=float, default=0.03)
    parser.add_argument("--tree_early_stopping_rounds", type=int, default=50)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_experiment(args)
