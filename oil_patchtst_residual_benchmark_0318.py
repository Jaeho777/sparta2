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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset


"""
Academic-style univariate benchmark for weekly oil forecasting.

Design choices
--------------
1. No exogenous variables and no shift(1) features.
2. PatchTST is the shared univariate baseline for both targets.
3. Residual correction models operate on the historical baseline residual series:
   PatchTST one-step fitted residuals -> NLinear / XGBoost / LightGBM.
4. Evaluation follows a common expanding-window ts-cv protocol plus final holdout.

Target mapping
--------------
- WTI Oil   -> Com_CrudeOil
- Brent Oil -> Com_BrentCrudeOil

This mapping is an explicit working assumption based on the current dataset schema.
"""


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
    batch_size: int = 32
    patience: int = 20
    weight_decay: float = 1e-5
    scaler_type: str = "identity"


@dataclass(frozen=True)
class NLinearConfig:
    max_steps: int = 2000
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
    target_range = float(np.max(y_true) - np.min(y_true))
    if target_range <= 1e-12:
        return float("nan")
    return rmse(y_true, y_pred) / target_range


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "NRMSE": nrmse_range(y_true, y_pred),
    }


def make_direct_windows(series: np.ndarray, input_size: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for end in range(input_size, len(series) - horizon + 1):
        xs.append(series[end - input_size : end])
        ys.append(series[end : end + horizon])
    if not xs:
        raise ValueError("Not enough observations to build direct multi-step windows.")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def make_one_step_windows(series: np.ndarray, input_size: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for end in range(input_size, len(series)):
        xs.append(series[end - input_size : end])
        ys.append(series[end])
    if not xs:
        raise ValueError("Not enough observations to build one-step windows.")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def split_timewise(
    x: np.ndarray,
    y: np.ndarray,
    min_val: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(x) < 3:
        raise ValueError("Need at least 3 samples for time-based train/validation split.")
    val_size = min(max(min_val, len(x) // 5), len(x) - 1)
    split_idx = len(x) - val_size
    return (
        x[:split_idx].copy(),
        y[:split_idx].copy(),
        x[split_idx:].copy(),
        y[split_idx:].copy(),
    )


class PatchTSTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        attn_dropout: float,
    ):
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


class PatchTSTForecaster(nn.Module):
    def __init__(self, protocol: ProtocolConfig, config: PatchTSTConfig):
        super().__init__()
        if protocol.input_size < config.patch_len:
            raise ValueError("patch_len must be <= input_size")
        n_patches = (protocol.input_size - config.patch_len) // config.stride + 1
        if n_patches <= 0:
            raise ValueError("Invalid patch configuration.")

        self.input_size = protocol.input_size
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
            nn.Linear(config.linear_hidden_size, protocol.horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(-1)
        patches = []
        for start in range(0, self.input_size - self.patch_len + 1, self.stride):
            patches.append(x[:, start : start + self.patch_len])
        z = torch.stack(patches, dim=1)
        z = self.patch_proj(z)
        z = self.input_dropout(z + self.pos_embedding)
        for block in self.encoder:
            z = block(z)
        return self.head(z)


class NLinearResidual(nn.Module):
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1:]
        x = x - last
        return self.linear(x) + last


def to_device_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
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
    train_loader = to_device_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = to_device_loader(x_val, y_val, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    model = model.to(device)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    steps = 0
    bad_epochs = 0

    while steps < max_steps and bad_epochs < patience:
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def predict_torch(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def fit_patchtst_baseline(
    train_series: np.ndarray,
    protocol: ProtocolConfig,
    config: PatchTSTConfig,
    device: torch.device,
) -> tuple[PatchTSTForecaster, dict[str, float | int]]:
    x, y = make_direct_windows(train_series, protocol.input_size, protocol.horizon)
    x = x[:, :, None]
    model = PatchTSTForecaster(protocol=protocol, config=config)
    model, fit_info = fit_torch_model(
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
    return model, fit_info


def forecast_patchtst(
    model: PatchTSTForecaster,
    history: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    x = history.astype(np.float32)[None, :, None]
    pred = predict_torch(model, x, batch_size=batch_size, device=device)
    return pred[0]


def compute_residual_history(
    model: PatchTSTForecaster,
    train_series: np.ndarray,
    protocol: ProtocolConfig,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    x, y = make_one_step_windows(train_series, protocol.input_size)
    preds = predict_torch(model, x[:, :, None], batch_size=batch_size, device=device)[:, 0]
    residuals = y - preds
    return residuals.astype(np.float32)


def fit_nlinear_residual(
    residual_series: np.ndarray,
    protocol: ProtocolConfig,
    config: NLinearConfig,
    device: torch.device,
) -> tuple[NLinearResidual, dict[str, float | int]]:
    x, y = make_direct_windows(residual_series, protocol.input_size, protocol.horizon)
    model = NLinearResidual(seq_len=protocol.input_size, pred_len=protocol.horizon)
    model, fit_info = fit_torch_model(
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
    return model, fit_info


def forecast_nlinear_residual(
    model: NLinearResidual,
    residual_history: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    x = residual_history.astype(np.float32)[None, :]
    pred = predict_torch(model, x, batch_size=batch_size, device=device)
    return pred[0]


def fit_tree_family(
    x: np.ndarray,
    y: np.ndarray,
    family: str,
    config: TreeConfig,
    seed: int,
) -> tuple[list[object], dict[str, int]]:
    x_train, y_train, x_val, y_val = split_timewise(x, y, min_val=12)
    x_train = np.ascontiguousarray(x_train)
    x_val = np.ascontiguousarray(x_val)
    models: list[object] = []

    for horizon_idx in range(y.shape[1]):
        target_train = np.ascontiguousarray(y_train[:, horizon_idx])
        target_val = np.ascontiguousarray(y_val[:, horizon_idx])

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
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(
                x_train,
                target_train,
                eval_set=[(x_val, target_val)],
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
            model.fit(
                x_train,
                target_train,
                eval_set=[(x_val, target_val)],
                verbose=False,
            )
        else:
            raise ValueError(f"Unsupported tree family: {family}")

        models.append(model)

    return models, {
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "n_models": int(len(models)),
    }


def fit_lgbm_residual(
    residual_series: np.ndarray,
    protocol: ProtocolConfig,
    config: TreeConfig,
    seed: int,
) -> tuple[list[object], dict[str, int]]:
    x, y = make_direct_windows(residual_series, protocol.input_size, protocol.horizon)
    return fit_tree_family(x=x, y=y, family="lgbm", config=config, seed=seed)


def fit_xgb_residual(
    residual_series: np.ndarray,
    protocol: ProtocolConfig,
    config: TreeConfig,
    seed: int,
) -> tuple[list[object], dict[str, int]]:
    x, y = make_direct_windows(residual_series, protocol.input_size, protocol.horizon)
    return fit_tree_family(x=x, y=y, family="xgb", config=config, seed=seed)


def forecast_tree_residual(models: list[object], residual_history: np.ndarray) -> np.ndarray:
    x = residual_history.astype(np.float32)[None, :]
    preds = []
    for model in models:
        preds.append(float(model.predict(x)[0]))
    return np.asarray(preds, dtype=np.float32)


def build_eval_windows(series_length: int, protocol: ProtocolConfig) -> tuple[list[EvalWindow], int]:
    ts_cv_span = protocol.horizon + protocol.step_size * (protocol.n_windows - 1)
    eval_span = ts_cv_span + protocol.final_holdout
    initial_train = series_length - eval_span

    min_required = protocol.input_size * 2 + protocol.horizon
    if initial_train < min_required:
        raise ValueError(
            "Protocol leaves too little training data. "
            f"initial_train={initial_train}, min_required={min_required}"
        )

    windows: list[EvalWindow] = []
    for idx in range(protocol.n_windows):
        train_end = initial_train + idx * protocol.step_size
        test_start = train_end
        test_end = test_start + protocol.horizon
        windows.append(
            EvalWindow(
                split="tscv",
                window_id=idx + 1,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    holdout_start = series_length - protocol.final_holdout
    windows.append(
        EvalWindow(
            split="holdout",
            window_id=1,
            train_end=holdout_start,
            test_start=holdout_start,
            test_end=series_length,
        )
    )
    return windows, initial_train


def run_one_window(
    target_name: str,
    target_col: str,
    series: pd.Series,
    window: EvalWindow,
    protocol: ProtocolConfig,
    patch_cfg: PatchTSTConfig,
    nlinear_cfg: NLinearConfig,
    tree_cfg: TreeConfig,
    device: torch.device,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    train_series = series.iloc[: window.train_end].to_numpy(dtype=np.float32)
    test_series = series.iloc[window.test_start : window.test_end].to_numpy(dtype=np.float32)
    test_dates = series.index[window.test_start : window.test_end]

    patch_model, patch_fit = fit_patchtst_baseline(
        train_series=train_series,
        protocol=protocol,
        config=patch_cfg,
        device=device,
    )
    baseline_forecast = forecast_patchtst(
        model=patch_model,
        history=train_series[-protocol.input_size :],
        batch_size=patch_cfg.batch_size,
        device=device,
    )

    residual_history = compute_residual_history(
        model=patch_model,
        train_series=train_series,
        protocol=protocol,
        batch_size=patch_cfg.batch_size,
        device=device,
    )
    residual_input = residual_history[-protocol.input_size :]

    nlinear_model, nlinear_fit = fit_nlinear_residual(
        residual_series=residual_history,
        protocol=protocol,
        config=nlinear_cfg,
        device=device,
    )
    nlinear_resid = forecast_nlinear_residual(
        model=nlinear_model,
        residual_history=residual_input,
        batch_size=nlinear_cfg.batch_size,
        device=device,
    )

    xgb_models, xgb_fit = fit_xgb_residual(
        residual_series=residual_history,
        protocol=protocol,
        config=tree_cfg,
        seed=seed,
    )
    xgb_resid = forecast_tree_residual(xgb_models, residual_input)

    lgbm_models, lgbm_fit = fit_lgbm_residual(
        residual_series=residual_history,
        protocol=protocol,
        config=tree_cfg,
        seed=seed,
    )
    lgbm_resid = forecast_tree_residual(lgbm_models, residual_input)

    model_predictions = [
        ("PatchTST", "-", baseline_forecast),
        ("PatchTST", "NLinear", baseline_forecast + nlinear_resid),
        ("PatchTST", "XGBoost", baseline_forecast + xgb_resid),
        ("PatchTST", "LightGBM", baseline_forecast + lgbm_resid),
    ]

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for baseline_name, residual_name, pred in model_predictions:
        metrics = compute_metrics(test_series, pred)
        metric_rows.append(
            {
                "Target": target_name,
                "TargetColumn": target_col,
                "EvalSplit": window.split,
                "WindowId": window.window_id,
                "TrainEndDate": str(series.index[window.train_end - 1].date()),
                "TestStartDate": str(series.index[window.test_start].date()),
                "TestEndDate": str(series.index[window.test_end - 1].date()),
                "TrainSize": int(window.train_end),
                "TestSize": int(len(test_series)),
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
                    "HorizonStep": step_idx + 1,
                    "Baseline": baseline_name,
                    "ResidualModel": residual_name,
                    "Actual": float(test_series[step_idx]),
                    "Prediction": float(pred[step_idx]),
                    "Error": float(test_series[step_idx] - pred[step_idx]),
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
        "ResidualHistoryLength": int(len(residual_history)),
    }
    return metric_rows, prediction_rows, fit_meta


def aggregate_leaderboard(prediction_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["Target", "TargetColumn", "EvalSplit", "Baseline", "ResidualModel"]
    for keys, group in prediction_df.groupby(group_cols, sort=False):
        target, target_col, eval_split, baseline, residual = keys
        metrics = compute_metrics(
            y_true=group["Actual"].to_numpy(dtype=float),
            y_pred=group["Prediction"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "Target": target,
                "TargetColumn": target_col,
                "EvalSplit": eval_split,
                "Baseline": baseline,
                "ResidualModel": residual,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MAPE": metrics["MAPE"],
                "NRMSE": metrics["NRMSE"],
                "NumForecastPoints": int(len(group)),
                "NumWindows": int(group["WindowId"].nunique()),
            }
        )
    leaderboard = pd.DataFrame(rows)
    return leaderboard.sort_values(["EvalSplit", "Target", "RMSE"]).reset_index(drop=True)


def resolve_target_selection(requested: list[str] | None) -> dict[str, str]:
    if not requested or requested == ["all"]:
        return TARGET_MAP

    normalized = {key.lower(): key for key in TARGET_MAP}
    selected: dict[str, str] = {}
    for item in requested:
        key = item.lower()
        if key in normalized:
            label = normalized[key]
            selected[label] = TARGET_MAP[label]
            continue
        if item in TARGET_MAP.values():
            label = next(name for name, col in TARGET_MAP.items() if col == item)
            selected[label] = item
            continue
        raise ValueError(f"Unknown target selection: {item}")
    return selected


def run_experiment(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = ProtocolConfig(
        input_size=args.input_size,
        horizon=args.horizon,
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

    selected_targets = resolve_target_selection(args.targets)
    missing = [col for col in selected_targets.values() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns in data: {missing}")

    windows, initial_train = build_eval_windows(series_length=len(df), protocol=protocol)

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    fit_meta_rows: list[dict[str, object]] = []

    print("=" * 80)
    print("PatchTST Residual Benchmark 0318")
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
        series = df[target_col].astype(np.float32)
        print(f"[Target] {target_name} ({target_col})")
        for window in windows:
            print(
                f"  - {window.split} window {window.window_id:02d}: "
                f"train_end={series.index[window.train_end - 1].date()}, "
                f"test={series.index[window.test_start].date()} ~ {series.index[window.test_end - 1].date()}"
            )
            window_metrics, window_predictions, fit_meta = run_one_window(
                target_name=target_name,
                target_col=target_col,
                series=series,
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
        "fit_metadata": fit_meta_rows,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Saved:")
    print(f"  - {out_dir / 'leaderboard_tscv.csv'}")
    print(f"  - {out_dir / 'leaderboard_holdout.csv'}")
    print(f"  - {out_dir / 'window_metrics.csv'}")
    print(f"  - {out_dir / 'window_predictions.csv'}")
    print(f"  - {out_dir / 'run_metadata.json'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PatchTST + residual correction benchmark for oil targets.")
    parser.add_argument("--data_path", default="data_weekly_260120.csv")
    parser.add_argument("--output_dir", default="output_oil_patchtst_residual_0318")
    parser.add_argument("--targets", nargs="*", default=["all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)

    parser.add_argument("--input_size", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--step_size", type=int, default=4)
    parser.add_argument("--n_windows", type=int, default=24)
    parser.add_argument("--final_holdout", type=int, default=12)
    parser.add_argument("--season_length", type=int, default=52)

    parser.add_argument("--patch_max_steps", type=int, default=5000)
    parser.add_argument("--patch_learning_rate", type=float, default=1e-4)
    parser.add_argument("--patch_batch_size", type=int, default=32)
    parser.add_argument("--patch_patience", type=int, default=20)

    parser.add_argument("--nlinear_max_steps", type=int, default=2000)
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
