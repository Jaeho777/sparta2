from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_codex"))
warnings.filterwarnings(
    "ignore",
    message="Usage of np.ndarray subset \\(sliced data\\) is not recommended.*",
    module="lightgbm",
)

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
from torch.utils.data import DataLoader, TensorDataset


TARGET_MAP = {
    "WTI Oil": "Com_CrudeOil",
    "Brent Oil": "Com_BrentCrudeOil",
}


@dataclass(frozen=True)
class ProtocolConfig:
    input_size: int = 48
    horizon: int = 12
    step_size: int = 4
    n_windows: int = 24
    final_holdout: int = 12
    season_length: int = 52
    inner_n_windows: int = 16
    inner_step_size: int = 4


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
    max_steps: int = 5000
    learning_rate: float = 1e-4
    scaler_type: str = "identity"
    batch_size: int = 32
    patience: int = 12
    weight_decay: float = 1e-5


@dataclass(frozen=True)
class NLinearConfig:
    hidden_size: int = 128
    max_steps: int = 300
    learning_rate: float = 1e-3
    batch_size: int = 16
    patience: int = 20
    weight_decay: float = 0.0


@dataclass(frozen=True)
class TreeConfig:
    n_estimators: int = 200
    learning_rate: float = 0.03
    max_depth: int = 4
    num_leaves: int = 31
    min_child_samples: int = 4
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 20


@dataclass(frozen=True)
class EvalWindow:
    split: str
    window_id: int
    train_end: int
    test_start: int
    test_end: int


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


class PatchTSTDirect(nn.Module):
    def __init__(self, input_size: int, horizon: int, config: PatchTSTConfig):
        super().__init__()
        n_patches = (input_size - config.patch_len) // config.stride + 1
        if n_patches <= 0:
            raise ValueError("Invalid patch configuration.")
        self.input_size = input_size
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.patch_proj = nn.Linear(config.patch_len, config.hidden_size)
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
            nn.Linear(config.linear_hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(-1)
        batch = x.size(0)
        patches = []
        for start in range(0, self.input_size - self.patch_len + 1, self.stride):
            patches.append(x[:, start : start + self.patch_len])
        z = torch.stack(patches, dim=1).reshape(batch, len(patches), self.patch_len)
        z = self.patch_proj(z)
        z = self.input_dropout(z + self.pos_embedding)
        for block in self.encoder:
            z = block(z)
        return self.head(z)


class NLinearDirect(nn.Module):
    def __init__(self, input_size: int, horizon: int, hidden_size: int):
        super().__init__()
        self.in_proj = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(-1)
        x_norm = x - x.mean(dim=1, keepdim=True)
        hidden = torch.relu(self.in_proj(x_norm))
        return self.out_proj(self.dropout(hidden))


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


def nrmse_fixed_range(y_true: np.ndarray, y_pred: np.ndarray, target_range: float) -> float:
    if target_range <= 1e-12:
        return float("nan")
    return rmse(y_true, y_pred) / target_range


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_range: float) -> dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "NRMSE": nrmse_fixed_range(y_true, y_pred, target_range),
    }


def slugify(text: str) -> str:
    return text.lower().replace(" ", "_").replace("/", "_").replace("-", "_").replace("+", "_plus_")


def model_label(baseline: str, residual: str) -> str:
    return baseline if residual == "-" else f"{baseline} + {residual}"


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
    x_train, y_train, x_val, y_val = split_timewise(x, y, min_val=max(4, min(12, len(x) // 4)))
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
                val_losses.append(criterion(model(xb), yb).item())
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


def predict_torch(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def make_direct_sequences(
    y: np.ndarray,
    input_size: int,
    horizon: int,
    end_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    origins = []
    for origin in range(input_size, end_idx - horizon + 1):
        xs.append(y[origin - input_size : origin].reshape(input_size, 1))
        ys.append(y[origin : origin + horizon])
        origins.append(origin)
    if not xs:
        raise ValueError("Not enough samples to build direct sequences.")
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(origins, dtype=np.int32),
    )


def fit_patchtst_direct(
    y_train: np.ndarray,
    input_size: int,
    horizon: int,
    config: PatchTSTConfig,
    device: torch.device,
) -> tuple[PatchTSTDirect, dict[str, float | int]]:
    x_seq, y_seq, _ = make_direct_sequences(y_train, input_size=input_size, horizon=horizon, end_idx=len(y_train))
    if config.scaler_type != "identity":
        raise ValueError(f"Unsupported scaler_type: {config.scaler_type}")

    model = PatchTSTDirect(input_size=input_size, horizon=horizon, config=config)
    model, fit_info = fit_torch_model(
        model=model,
        x=x_seq,
        y=y_seq.astype(np.float32),
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        patience=config.patience,
        weight_decay=config.weight_decay,
        device=device,
    )
    return model, fit_info


def predict_patchtst_direct(
    model: PatchTSTDirect,
    history: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    x = history.astype(np.float32).reshape(1, -1, 1)
    return predict_torch(model, x, batch_size=batch_size, device=device)[0]


def fit_nlinear_direct(
    x_train: np.ndarray,
    y_train: np.ndarray,
    input_size: int,
    horizon: int,
    config: NLinearConfig,
    device: torch.device,
) -> tuple[NLinearDirect, dict[str, float | int]]:
    model = NLinearDirect(input_size=input_size, horizon=horizon, hidden_size=config.hidden_size)
    model, fit_info = fit_torch_model(
        model=model,
        x=x_train,
        y=y_train.astype(np.float32),
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        patience=config.patience,
        weight_decay=config.weight_decay,
        device=device,
    )
    return model, fit_info


def fit_tree_direct(
    x_train: np.ndarray,
    y_train: np.ndarray,
    family: str,
    config: TreeConfig,
    seed: int,
) -> tuple[list[object], dict[str, int]]:
    use_validation = len(x_train) >= 6
    if use_validation:
        x_fit, y_fit, x_val, y_val = split_timewise(x_train, y_train, min_val=max(2, min(4, len(x_train) // 3)))
    else:
        x_fit = x_train.copy()
        y_fit = y_train.copy()
        x_val = np.empty((0, x_train.shape[1]), dtype=np.float32)
        y_val = np.empty((0, y_train.shape[1]), dtype=np.float32)
    models = []
    for horizon_idx in range(y_train.shape[1]):
        target_fit = y_fit[:, horizon_idx]
        target_val = y_val[:, horizon_idx] if use_validation else None
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
                random_state=seed + horizon_idx,
                verbosity=-1,
                n_jobs=-1,
            )
            fit_kwargs = {}
            if use_validation:
                fit_kwargs = {
                    "eval_set": [(x_val, target_val)],
                    "eval_metric": "l2",
                    "callbacks": [lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
                }
            model.fit(x_fit, target_fit, **fit_kwargs)
        elif family == "xgb":
            model_kwargs = dict(
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                max_depth=config.max_depth,
                min_child_weight=config.min_child_weight,
                subsample=config.subsample,
                colsample_bytree=config.colsample_bytree,
                reg_alpha=config.reg_alpha,
                reg_lambda=config.reg_lambda,
                objective="reg:squarederror",
                random_state=seed + horizon_idx,
                n_jobs=-1,
                tree_method="hist",
                verbosity=0,
            )
            if use_validation:
                model_kwargs["early_stopping_rounds"] = config.early_stopping_rounds
            model = xgb.XGBRegressor(**model_kwargs)
            fit_kwargs = {"verbose": False}
            if use_validation:
                fit_kwargs["eval_set"] = [(x_val, target_val)]
            model.fit(x_fit, target_fit, **fit_kwargs)
        else:
            raise ValueError(f"Unsupported family: {family}")
        models.append(model)
    return models, {
        "train_samples": int(len(x_fit)),
        "val_samples": int(len(x_val)),
    }


def predict_tree_direct(models: list[object], x: np.ndarray) -> np.ndarray:
    preds = [float(model.predict(x)[0]) for model in models]
    return np.asarray(preds, dtype=np.float32)


def build_outer_windows(series_length: int, protocol: ProtocolConfig) -> tuple[list[EvalWindow], int]:
    ts_cv_span = protocol.horizon + protocol.step_size * (protocol.n_windows - 1)
    eval_span = ts_cv_span + protocol.final_holdout
    initial_train = series_length - eval_span
    if initial_train < protocol.input_size + protocol.horizon:
        raise ValueError("Initial training segment is too short.")

    windows = []
    for idx in range(protocol.n_windows):
        train_end = initial_train + idx * protocol.step_size
        test_start = train_end
        test_end = test_start + protocol.horizon
        windows.append(EvalWindow("tscv", idx + 1, train_end, test_start, test_end))

    holdout_start = series_length - protocol.final_holdout
    windows.append(EvalWindow("holdout", 1, holdout_start, holdout_start, series_length))
    return windows, initial_train


def generate_inner_oof_residuals(
    y_train: np.ndarray,
    dates: pd.DatetimeIndex,
    protocol: ProtocolConfig,
    patch_cfg: PatchTSTConfig,
    device: torch.device,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[dict[str, object]]]:
    inner_span = protocol.horizon + protocol.inner_step_size * (protocol.inner_n_windows - 1)
    inner_initial_train = len(y_train) - inner_span
    if inner_initial_train < protocol.input_size + protocol.horizon:
        raise ValueError("Training window too short for inner OOF residuals.")

    x_rows = []
    y_rows = []
    residual_rows = []
    fit_meta = []

    for idx in range(protocol.inner_n_windows):
        train_end = inner_initial_train + idx * protocol.inner_step_size
        test_start = train_end
        test_end = test_start + protocol.horizon
        set_seed(base_seed + idx + 1)
        model, fit_info = fit_patchtst_direct(
            y_train=y_train[:train_end],
            input_size=protocol.input_size,
            horizon=protocol.horizon,
            config=patch_cfg,
            device=device,
        )
        history = y_train[train_end - protocol.input_size : train_end]
        actual = y_train[test_start:test_end]
        pred = predict_patchtst_direct(
            model=model,
            history=history,
            batch_size=patch_cfg.batch_size,
            device=device,
        )
        residual = actual - pred
        x_rows.append(history.reshape(protocol.input_size, 1))
        y_rows.append(residual.astype(np.float32))
        for step in range(protocol.horizon):
            residual_rows.append(
                {
                    "InnerWindowId": idx + 1,
                    "ForecastStartDate": str(dates[test_start].date()),
                    "Date": str(dates[test_start + step].date()),
                    "HorizonStep": int(step + 1),
                    "Residual": float(residual[step]),
                    "Actual": float(actual[step]),
                    "BasePrediction": float(pred[step]),
                }
            )
        fit_meta.append(
            {
                "InnerWindowId": idx + 1,
                "TrainSize": int(train_end),
                "TestStartDate": str(dates[test_start].date()),
                "TestEndDate": str(dates[test_end - 1].date()),
                **fit_info,
            }
        )

    return (
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.float32),
        pd.DataFrame(residual_rows),
        fit_meta,
    )


def run_one_window(
    target_name: str,
    target_col: str,
    y_all: np.ndarray,
    dates: pd.DatetimeIndex,
    window: EvalWindow,
    protocol: ProtocolConfig,
    patch_cfg: PatchTSTConfig,
    nlinear_cfg: NLinearConfig,
    tree_cfg: TreeConfig,
    target_range: float,
    device: torch.device,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    target_seed_offset = 0 if target_name == "WTI Oil" else 10000
    split_seed_offset = 0 if window.split == "tscv" else 50000
    base_seed = seed + target_seed_offset + split_seed_offset + window.window_id * 100

    train_end = window.train_end
    test_start = window.test_start
    test_end = window.test_end

    y_train = y_all[:train_end]
    history = y_all[train_end - protocol.input_size : train_end]
    actual = y_all[test_start:test_end]
    test_dates = dates[test_start:test_end]

    set_seed(base_seed)
    patch_model, patch_fit = fit_patchtst_direct(
        y_train=y_train,
        input_size=protocol.input_size,
        horizon=protocol.horizon,
        config=patch_cfg,
        device=device,
    )
    baseline_pred = predict_patchtst_direct(
        model=patch_model,
        history=history,
        batch_size=patch_cfg.batch_size,
        device=device,
    )

    x_resid_train, y_resid_train, inner_residual_df, inner_fit_meta = generate_inner_oof_residuals(
        y_train=y_train,
        dates=dates[:train_end],
        protocol=protocol,
        patch_cfg=patch_cfg,
        device=device,
        base_seed=base_seed + 1000,
    )
    x_resid_test = history.astype(np.float32).reshape(1, protocol.input_size, 1)

    set_seed(base_seed + 1)
    nlinear_model, nlinear_fit = fit_nlinear_direct(
        x_train=x_resid_train,
        y_train=y_resid_train,
        input_size=protocol.input_size,
        horizon=protocol.horizon,
        config=nlinear_cfg,
        device=device,
    )
    nlinear_resid = predict_torch(nlinear_model, x_resid_test, batch_size=nlinear_cfg.batch_size, device=device)[0]

    x_flat = x_resid_train.reshape(len(x_resid_train), -1)
    x_flat_test = x_resid_test.reshape(1, -1)

    xgb_models, xgb_fit = fit_tree_direct(
        x_train=x_flat,
        y_train=y_resid_train,
        family="xgb",
        config=tree_cfg,
        seed=base_seed + 2,
    )
    xgb_resid = predict_tree_direct(xgb_models, x_flat_test)

    lgbm_models, lgbm_fit = fit_tree_direct(
        x_train=x_flat,
        y_train=y_resid_train,
        family="lgbm",
        config=tree_cfg,
        seed=base_seed + 3,
    )
    lgbm_resid = predict_tree_direct(lgbm_models, x_flat_test)

    predictions = [
        ("PatchTST", "-", baseline_pred),
        ("PatchTST", "NLinear", baseline_pred + nlinear_resid),
        ("PatchTST", "XGBoost", baseline_pred + xgb_resid),
        ("PatchTST", "LightGBM", baseline_pred + lgbm_resid),
    ]

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for baseline_name, residual_name, pred in predictions:
        metrics = compute_metrics(actual, pred, target_range=target_range)
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
                "TestSize": int(len(actual)),
                "Baseline": baseline_name,
                "ResidualModel": residual_name,
                **metrics,
            }
        )
        for step_idx, dt in enumerate(test_dates):
            prediction_rows.append(
                {
                    "Target": target_name,
                    "TargetColumn": target_col,
                    "EvalSplit": window.split,
                    "WindowId": window.window_id,
                    "Date": str(dt.date()),
                    "RawPosition": int(test_start + step_idx),
                    "HorizonStep": int(step_idx + 1),
                    "Baseline": baseline_name,
                    "ResidualModel": residual_name,
                    "Actual": float(actual[step_idx]),
                    "Prediction": float(pred[step_idx]),
                    "Error": float(actual[step_idx] - pred[step_idx]),
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
        "InnerWindows": int(protocol.inner_n_windows),
        "InnerFitMeta": inner_fit_meta,
        "InnerResidualSummary": {
            "rows": int(len(inner_residual_df)),
            "forecast_starts": int(inner_residual_df["ForecastStartDate"].nunique()),
            "unique_dates": int(inner_residual_df["Date"].nunique()),
        },
    }
    return metric_rows, prediction_rows, fit_meta


def aggregate_predictions_dateavg(prediction_df: pd.DataFrame) -> pd.DataFrame:
    return (
        prediction_df.groupby(
            ["Target", "TargetColumn", "EvalSplit", "Baseline", "ResidualModel", "Date"],
            as_index=False,
        )[["Actual", "Prediction"]]
        .mean()
        .sort_values(["Target", "EvalSplit", "Date", "Baseline", "ResidualModel"])
        .reset_index(drop=True)
    )


def aggregate_leaderboard(dateavg_df: pd.DataFrame, target_ranges: dict[str, float]) -> pd.DataFrame:
    rows = []
    group_cols = ["Target", "TargetColumn", "EvalSplit", "Baseline", "ResidualModel"]
    for keys, group in dateavg_df.groupby(group_cols, sort=False):
        target, target_col, eval_split, baseline, residual = keys
        metrics = compute_metrics(
            group["Actual"].to_numpy(dtype=float),
            group["Prediction"].to_numpy(dtype=float),
            target_ranges[target],
        )
        rows.append(
            {
                "Target": target,
                "TargetColumn": target_col,
                "EvalSplit": eval_split,
                "Baseline": baseline,
                "ResidualModel": residual,
                **metrics,
                "NumDates": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["EvalSplit", "Target", "RMSE"]).reset_index(drop=True)


def save_prediction_plots(dateavg_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    color_map = {
        "PatchTST": "#7f7f7f",
        "PatchTST + NLinear": "#d62728",
        "PatchTST + XGBoost": "#1f77b4",
        "PatchTST + LightGBM": "#2ca02c",
    }
    plot_rows = []
    for (eval_split, target), group in dateavg_df.groupby(["EvalSplit", "Target"], sort=False):
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
                lw=2.0,
                label=label,
                color=color_map.get(label),
            )

        ax.set_title(f"{target} | {eval_split.upper()} | Actual vs Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", ncol=2)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        fig.tight_layout()

        filename = f"{slugify(eval_split)}_{slugify(target)}_actual_vs_pred.png"
        path = plot_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)

        plot_rows.append(
            {
                "EvalSplit": eval_split,
                "Target": target,
                "PlotPath": str(path),
            }
        )
    return pd.DataFrame(plot_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict univariate horizon-12 PatchTST residual benchmark.")
    parser.add_argument("--data_path", type=Path, default=Path("data_weekly_260120.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("output_oil_univariate_direct_residual_0320"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_max_steps", type=int, default=5000)
    parser.add_argument("--nlinear_max_steps", type=int, default=300)
    parser.add_argument("--tree_n_estimators", type=int, default=200)
    parser.add_argument("--tree_early_stopping_rounds", type=int, default=20)
    parser.add_argument("--n_windows", type=int, default=24)
    parser.add_argument("--inner_n_windows", type=int, default=16)
    parser.add_argument("--targets", nargs="*", default=list(TARGET_MAP.keys()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    protocol = ProtocolConfig(
        n_windows=args.n_windows,
        inner_n_windows=args.inner_n_windows,
    )
    patch_cfg = PatchTSTConfig(max_steps=args.patch_max_steps)
    nlinear_cfg = NLinearConfig(max_steps=args.nlinear_max_steps)
    tree_cfg = TreeConfig(
        n_estimators=args.tree_n_estimators,
        early_stopping_rounds=args.tree_early_stopping_rounds,
    )

    df = pd.read_csv(args.data_path)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True).set_index("dt")

    windows, initial_train = build_outer_windows(len(df), protocol)

    metric_rows = []
    prediction_rows = []
    fit_meta = []

    target_ranges = {
        target_name: float(df[target_col].max() - df[target_col].min())
        for target_name, target_col in TARGET_MAP.items()
    }

    for target_name in args.targets:
        target_col = TARGET_MAP[target_name]
        y_all = df[target_col].to_numpy(dtype=np.float32)
        for window in windows:
            print(
                f"[run] target={target_name} split={window.split} window={window.window_id}",
                flush=True,
            )
            window_metrics, window_preds, window_fit = run_one_window(
                target_name=target_name,
                target_col=target_col,
                y_all=y_all,
                dates=df.index,
                window=window,
                protocol=protocol,
                patch_cfg=patch_cfg,
                nlinear_cfg=nlinear_cfg,
                tree_cfg=tree_cfg,
                target_range=target_ranges[target_name],
                device=device,
                seed=args.seed,
            )
            metric_rows.extend(window_metrics)
            prediction_rows.extend(window_preds)
            fit_meta.append(window_fit)

    metric_df = pd.DataFrame(metric_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    dateavg_df = aggregate_predictions_dateavg(prediction_df)
    leaderboard_df = aggregate_leaderboard(dateavg_df, target_ranges=target_ranges)
    plot_manifest = save_prediction_plots(dateavg_df, args.output_dir)

    metric_df.to_csv(args.output_dir / "window_metrics.csv", index=False)
    prediction_df.to_csv(args.output_dir / "window_predictions.csv", index=False)
    dateavg_df.to_csv(args.output_dir / "dateavg_predictions.csv", index=False)
    leaderboard_df.to_csv(args.output_dir / "leaderboard_all.csv", index=False)
    leaderboard_df[leaderboard_df["EvalSplit"] == "tscv"].to_csv(args.output_dir / "leaderboard_tscv.csv", index=False)
    leaderboard_df[leaderboard_df["EvalSplit"] == "holdout"].to_csv(args.output_dir / "leaderboard_holdout.csv", index=False)
    plot_manifest.to_csv(args.output_dir / "plot_manifest.csv", index=False)

    metadata = {
        "script": Path(__file__).name,
        "protocol": asdict(protocol),
        "patch_cfg": asdict(patch_cfg),
        "nlinear_cfg": asdict(nlinear_cfg),
        "tree_cfg": asdict(tree_cfg),
        "data_path": str(args.data_path.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "targets": args.targets,
        "series_length": int(len(df)),
        "data_start": str(df.index[0].date()),
        "data_end": str(df.index[-1].date()),
        "initial_train_size": int(initial_train),
        "initial_train_end_date": str(df.index[initial_train - 1].date()),
        "tscv_start_date": str(df.index[initial_train].date()),
        "tscv_end_date": str(df.index[initial_train + protocol.horizon + protocol.step_size * (protocol.n_windows - 1) - 1].date()),
        "holdout_start_date": str(df.index[len(df) - protocol.final_holdout].date()),
        "holdout_end_date": str(df.index[-1].date()),
        "fit_meta": fit_meta,
    }
    with open(args.output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
