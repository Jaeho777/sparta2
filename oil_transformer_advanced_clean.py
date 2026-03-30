"""
Leakage-conscious oil forecasting pipeline for professor review.

Deliverables
1. SHAP + TimeSeriesCV feature selection on train only.
2. Full 5x5 stage-1/stage-2 model grid over transformer/nonlinear models.
3. Top-3 selection by validation RMSE only.
4. Pure RoR strategies only. Strategy G is an RoR strategy, not RW blend.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset


DATA_PATH = "data_weekly_260120.csv"
TARGET = "Com_BrentCrudeOil"
VAL_START = "2025-08-04"
TEST_START = "2025-10-27"
OUT_DIR = "output_oil_transformer_clean"
SEED = 42
SEQ_LEN = 24
SCREEN_STAGE_SEEDS = 1
SCREEN_EPOCHS = 60
SCREEN_PATIENCE = 12
CONFIRM_TOPK = 8
CONFIRM_STAGE_SEEDS = 3
CONFIRM_EPOCHS = 120
CONFIRM_PATIENCE = 20

BASE_FEATURES = [
    "Com_Gasoline",
    "Com_NaturalGas",
    "Com_Uranium",
    "Com_Coal",
    "Com_LME_Cu_Cash",
    "Com_Steel",
    "Com_Iron_Ore",
    "Idx_DxyUSD",
    "EX_USD_CNY",
    "Bonds_US_10Y",
    "Bonds_US_2Y",
    "Bonds_US_3M",
    "Idx_SnPVIX",
    "Com_Gold",
    "Idx_SnP500",
    "Idx_CSI300",
    "EX_USD_KRW",
    "Bonds_KOR_10Y",
    "EX_USD_JPY",
    "Com_Corn",
    "Com_Soybeans",
    "Com_PalmOil",
]

MODEL_NAMES = [
    "NLinear",
    "PatchTST",
    "iTransformer",
    "Transformer",
    "LSTM",
]

EXPERIMENTS = [(base_name, resid_name) for base_name in MODEL_NAMES for resid_name in MODEL_NAMES]


@dataclass
class SequenceBundle:
    Xtr_seq: np.ndarray
    ytr_seq: np.ndarray
    Xva_seq: np.ndarray
    yva_seq: np.ndarray
    Xte_seq: np.ndarray
    yte_seq: np.ndarray
    Xtr_sc: np.ndarray
    Xva_sc: np.ndarray
    Xte_sc: np.ndarray
    Xtr_all_sc: np.ndarray
    Xva_all_sc: np.ndarray
    Xte_all_sc: np.ndarray
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    ymean: float
    ystd: float
    selected_features: list[str]
    all_features: list[str]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def nrmse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(rmse(y_true, y_pred) / np.mean(y_true) * 100.0)


def r2_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sst = float(np.mean((y_true - np.mean(y_true)) ** 2))
    if sst <= 1e-12:
        return float("nan")
    return float(1.0 - np.mean((y_true - y_pred) ** 2) / sst)


def safe_to_price(pred: np.ndarray, ymean: float, ystd: float) -> np.ndarray:
    return pred * ystd + ymean


def build_engineered_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    base = [c for c in BASE_FEATURES if c in df.columns]
    derived = pd.DataFrame(index=df.index)

    for col in base:
        derived[f"{col}_ret"] = df[col].pct_change()

    derived["Spread_10Y_2Y"] = df["Bonds_US_10Y"] - df["Bonds_US_2Y"]
    derived["Spread_Crack"] = df["Com_Gasoline"] - df[TARGET]
    derived["Ratio_Gold_Oil"] = df["Com_Gold"] / df[TARGET]

    for col in ["Com_Gasoline", "Com_NaturalGas", "Idx_SnPVIX", "Idx_DxyUSD"]:
        if col in df.columns:
            derived[f"{col}_ma4r"] = df[col] / df[col].rolling(4).mean() - 1.0
            derived[f"{col}_ma12r"] = df[col] / df[col].rolling(12).mean() - 1.0

    return derived, base


def select_features(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    lgb_shap = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        verbose=-1,
        random_state=SEED,
    )
    lgb_shap.fit(X_train_raw, y_train)
    explainer = shap.TreeExplainer(lgb_shap)
    shap_values = explainer.shap_values(X_train_raw)
    mean_shap = np.abs(shap_values).mean(axis=0)

    shap_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    ranked = shap_df["feature"].tolist()
    tscv = TimeSeriesSplit(n_splits=5)
    cv_rows = []

    for n_features in [10, 15, 20, 25, 30, 40, len(feature_names)]:
        n_features = min(n_features, len(feature_names))
        idx = [feature_names.index(f) for f in ranked[:n_features]]
        X_slice = X_train_raw[:, idx]
        fold_rmses = []

        for train_idx, val_idx in tscv.split(X_slice):
            fold_model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=20,
                verbose=-1,
                random_state=SEED,
            )
            fold_model.fit(X_slice[train_idx], y_train[train_idx])
            fold_pred = fold_model.predict(X_slice[val_idx])
            fold_rmses.append(rmse(y_train[val_idx], fold_pred))

        cv_rows.append(
            {
                "n_features": n_features,
                "cv_rmse": float(np.mean(fold_rmses)),
            }
        )

    cv_df = pd.DataFrame(cv_rows)
    best_n = int(cv_df.loc[cv_df["cv_rmse"].idxmin(), "n_features"])
    return ranked[:best_n], shap_df, cv_df


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len : i])
        ys.append(y[i])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def prepare_sequences(df: pd.DataFrame) -> SequenceBundle:
    derived, base = build_engineered_features(df)
    all_features = base + list(derived.columns)
    X_lag = pd.concat([df[base], derived], axis=1).shift(1)
    y = df[TARGET].copy()

    mask_train = df.index < VAL_START
    mask_val = (df.index >= VAL_START) & (df.index < TEST_START)
    mask_test = df.index >= TEST_START

    Xtr_raw = np.nan_to_num(X_lag.loc[mask_train].to_numpy(), nan=0.0)
    Xva_raw = np.nan_to_num(X_lag.loc[mask_val].to_numpy(), nan=0.0)
    Xte_raw = np.nan_to_num(X_lag.loc[mask_test].to_numpy(), nan=0.0)

    y_train = y.loc[mask_train]
    y_val = y.loc[mask_val]
    y_test = y.loc[mask_test]

    selected_features, shap_df, cv_df = select_features(
        Xtr_raw,
        y_train.to_numpy(dtype=float),
        all_features,
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    shap_df.to_csv(f"{OUT_DIR}/feature_shap.csv", index=False)
    cv_df.to_csv(f"{OUT_DIR}/feature_selection_cv.csv", index=False)

    selected_idx = [all_features.index(f) for f in selected_features]
    Xtr_sel = Xtr_raw[:, selected_idx]
    Xva_sel = Xva_raw[:, selected_idx]
    Xte_sel = Xte_raw[:, selected_idx]

    scaler_sel = RobustScaler()
    Xtr_sc = scaler_sel.fit_transform(Xtr_sel)
    Xva_sc = scaler_sel.transform(Xva_sel)
    Xte_sc = scaler_sel.transform(Xte_sel)

    scaler_all = RobustScaler()
    Xtr_all_sc = scaler_all.fit_transform(Xtr_raw)
    Xva_all_sc = scaler_all.transform(Xva_raw)
    Xte_all_sc = scaler_all.transform(Xte_raw)

    ymean = float(y_train.mean())
    ystd = float(y_train.std())
    ytr_n = ((y_train.to_numpy(dtype=float) - ymean) / ystd).astype(np.float32)
    yva_n = ((y_val.to_numpy(dtype=float) - ymean) / ystd).astype(np.float32)
    yte_n = ((y_test.to_numpy(dtype=float) - ymean) / ystd).astype(np.float32)

    Xtr_seq, ytr_seq = make_sequences(Xtr_sc, ytr_n, SEQ_LEN)

    val_X_buffer = Xtr_sc[-SEQ_LEN:]
    val_y_buffer = ytr_n[-SEQ_LEN:]
    Xva_seq, yva_seq = make_sequences(
        np.vstack([val_X_buffer, Xva_sc]),
        np.concatenate([val_y_buffer, yva_n]),
        SEQ_LEN,
    )

    test_X_buffer = np.vstack([Xtr_sc, Xva_sc])[-SEQ_LEN:]
    test_y_buffer = np.concatenate([ytr_n, yva_n])[-SEQ_LEN:]
    Xte_seq, yte_seq = make_sequences(
        np.vstack([test_X_buffer, Xte_sc]),
        np.concatenate([test_y_buffer, yte_n]),
        SEQ_LEN,
    )

    return SequenceBundle(
        Xtr_seq=Xtr_seq,
        ytr_seq=ytr_seq,
        Xva_seq=Xva_seq,
        yva_seq=yva_seq,
        Xte_seq=Xte_seq,
        yte_seq=yte_seq,
        Xtr_sc=Xtr_sc,
        Xva_sc=Xva_sc,
        Xte_sc=Xte_sc,
        Xtr_all_sc=Xtr_all_sc,
        Xva_all_sc=Xva_all_sc,
        Xte_all_sc=Xte_all_sc,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        ymean=ymean,
        ystd=ystd,
        selected_features=selected_features,
        all_features=all_features,
    )


class NLinearModel(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.time_proj = nn.Linear(seq_len, hidden)
        self.feature_proj = nn.Linear(n_features, hidden)
        self.output = nn.Linear(hidden * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_channel = x[:, :, 0]
        time_channel = time_channel - time_channel[:, -1:]
        h1 = self.act(self.time_proj(time_channel))
        h2 = self.act(self.feature_proj(x[:, -1, :]))
        return self.output(self.dropout(torch.cat([h1, h2], dim=-1))).squeeze(-1)


class PatchTSTModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        patch_len: int = 4,
        stride: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        n_patches = (seq_len - patch_len) // stride + 1
        self.proj = nn.Linear(patch_len * n_features, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        patches = []
        for start in range(0, x.size(1) - self.patch_len + 1, self.stride):
            patches.append(x[:, start : start + self.patch_len, :].reshape(batch, -1))
        z = self.proj(torch.stack(patches, dim=1))
        z = torch.cat([self.cls.expand(batch, -1, -1), z], dim=1) + self.pos
        z = self.encoder(z)
        return self.head(self.dropout(z[:, 0])).squeeze(-1)


class ITransformerModel(nn.Module):
    def __init__(self, seq_len: int, n_features: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(seq_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(
            nn.Linear(n_features * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x.permute(0, 2, 1)) + self.pos
        z = self.encoder(z).reshape(x.size(0), -1)
        return self.head(z).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self, seq_len: int, n_features: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x) + self.pos
        return self.head(self.encoder(z)[:, -1]).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def build_model(name: str, n_features: int) -> nn.Module:
    if name == "NLinear":
        return NLinearModel(SEQ_LEN, n_features)
    if name == "PatchTST":
        return PatchTSTModel(SEQ_LEN, n_features)
    if name == "iTransformer":
        return ITransformerModel(SEQ_LEN, n_features)
    if name == "Transformer":
        return TransformerModel(SEQ_LEN, n_features)
    if name == "LSTM":
        return LSTMModel(n_features)
    raise ValueError(f"Unknown model: {name}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    ystd: float,
    epochs: int = 150,
    patience: int = 25,
    lr: float = 1e-3,
) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    wait = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_model(model: nn.Module, X_seq: np.ndarray, device: torch.device) -> np.ndarray:
    if len(X_seq) == 0:
        return np.asarray([], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X_seq).to(device)).cpu().numpy()


def run_stage2_experiment(
    bundle: SequenceBundle,
    base_name: str,
    resid_name: str,
    device: torch.device,
    n_seeds: int,
    epochs: int,
    patience: int,
    keep_artifacts: bool,
) -> tuple[dict[str, float | str], dict[str, np.ndarray | str | float] | None]:
    label = f"{base_name}+{resid_name}"
    t0 = time.time()
    print(f"[Stage2] {label} | seeds={n_seeds} epochs={epochs} patience={patience}", flush=True)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(bundle.Xtr_seq), torch.from_numpy(bundle.ytr_seq)),
        batch_size=32,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(bundle.Xva_seq), torch.from_numpy(bundle.yva_seq)),
        batch_size=32,
        shuffle=False,
    )

    base_val_preds, base_test_preds, base_train_preds = [], [], []
    for seed_offset in range(n_seeds):
        set_seed(SEED + seed_offset)
        model = build_model(base_name, bundle.Xtr_seq.shape[2])
        model = train_model(
            model,
            train_loader,
            val_loader,
            device,
            bundle.ystd,
            epochs=epochs,
            patience=patience,
        )
        base_val_preds.append(predict_model(model, bundle.Xva_seq, device))
        base_test_preds.append(predict_model(model, bundle.Xte_seq, device))
        base_train_preds.append(predict_model(model, bundle.Xtr_seq, device))

    base_val = np.mean(base_val_preds, axis=0)
    base_test = np.mean(base_test_preds, axis=0)
    base_train = np.mean(base_train_preds, axis=0)

    base_val_price = safe_to_price(base_val, bundle.ymean, bundle.ystd)
    base_test_price = safe_to_price(base_test, bundle.ymean, bundle.ystd)
    y_val_arr = bundle.y_val.to_numpy(dtype=float)[-len(base_val_price) :]
    y_test_arr = bundle.y_test.to_numpy(dtype=float)[-len(base_test_price) :]
    base_val_rmse = rmse(y_val_arr, base_val_price)
    base_test_rmse = rmse(y_test_arr, base_test_price)
    base_test_mae = float(mean_absolute_error(y_test_arr, base_test_price))
    base_test_mape = mape(y_test_arr, base_test_price)
    base_val_nrmse = nrmse_pct(y_val_arr, base_val_price)
    base_val_r2 = r2_metric(y_val_arr, base_val_price)
    base_test_nrmse = nrmse_pct(y_test_arr, base_test_price)
    base_test_r2 = r2_metric(y_test_arr, base_test_price)

    resid_train_target = (bundle.ytr_seq - base_train).astype(np.float32)
    resid_val_target = (bundle.yva_seq - base_val[-len(bundle.yva_seq) :]).astype(np.float32)

    resid_train_loader = DataLoader(
        TensorDataset(torch.from_numpy(bundle.Xtr_seq), torch.from_numpy(resid_train_target)),
        batch_size=32,
        shuffle=True,
    )
    resid_val_loader = DataLoader(
        TensorDataset(torch.from_numpy(bundle.Xva_seq), torch.from_numpy(resid_val_target)),
        batch_size=32,
        shuffle=False,
    )

    resid_val_preds, resid_test_preds, resid_train_preds = [], [], []
    for seed_offset in range(n_seeds):
        set_seed(SEED + 100 + seed_offset)
        model = build_model(resid_name, bundle.Xtr_seq.shape[2])
        model = train_model(
            model,
            resid_train_loader,
            resid_val_loader,
            device,
            bundle.ystd,
            epochs=epochs,
            patience=patience,
        )
        resid_val_preds.append(predict_model(model, bundle.Xva_seq, device))
        resid_test_preds.append(predict_model(model, bundle.Xte_seq, device))
        if keep_artifacts:
            resid_train_preds.append(predict_model(model, bundle.Xtr_seq, device))

    resid_val = np.mean(resid_val_preds, axis=0)
    resid_test = np.mean(resid_test_preds, axis=0)
    resid_train = np.mean(resid_train_preds, axis=0) if keep_artifacts else None

    s2_val = safe_to_price(base_val[-len(resid_val) :] + resid_val, bundle.ymean, bundle.ystd)
    s2_test = safe_to_price(base_test[-len(resid_test) :] + resid_test, bundle.ymean, bundle.ystd)
    y_val_s2 = bundle.y_val.to_numpy(dtype=float)[-len(s2_val) :]
    y_test_s2 = bundle.y_test.to_numpy(dtype=float)[-len(s2_test) :]
    s2_val_rmse = rmse(y_val_s2, s2_val)
    s2_test_rmse = rmse(y_test_s2, s2_test)
    s2_test_mae = float(mean_absolute_error(y_test_s2, s2_test))
    s2_test_mape = mape(y_test_s2, s2_test)
    s2_val_nrmse = nrmse_pct(y_val_s2, s2_val)
    s2_val_r2 = r2_metric(y_val_s2, s2_val)
    s2_test_nrmse = nrmse_pct(y_test_s2, s2_test)
    s2_test_r2 = r2_metric(y_test_s2, s2_test)

    row = {
        "Experiment": label,
        "Base": base_name,
        "Residual": resid_name,
        "Base_Val_RMSE": base_val_rmse,
        "Base_Test_RMSE": base_test_rmse,
        "Base_Test_MAE": base_test_mae,
        "Base_Test_MAPE(%)": base_test_mape,
        "Base_Val_NRMSE(%)": base_val_nrmse,
        "Base_Val_R2": base_val_r2,
        "Base_Test_NRMSE(%)": base_test_nrmse,
        "Base_Test_R2": base_test_r2,
        "S2_Val_RMSE": s2_val_rmse,
        "S2_Test_RMSE": s2_test_rmse,
        "S2_Test_MAE": s2_test_mae,
        "S2_Test_MAPE(%)": s2_test_mape,
        "S2_Val_NRMSE(%)": s2_val_nrmse,
        "S2_Val_R2": s2_val_r2,
        "S2_Test_NRMSE(%)": s2_test_nrmse,
        "S2_Test_R2": s2_test_r2,
        "Seeds": n_seeds,
        "Epochs": epochs,
        "Patience": patience,
        "ElapsedSec": time.time() - t0,
    }

    if not keep_artifacts:
        return row, None

    artifacts = {
        "base_train": base_train,
        "base_val": base_val,
        "base_test": base_test,
        "resid_train": resid_train,
        "resid_val": resid_val,
        "resid_test": resid_test,
        "s2_val": s2_val,
        "s2_test": s2_test,
        "resid_train_target": resid_train_target,
    }
    return row, artifacts


def build_ror_augmented_features(
    X_flat: np.ndarray,
    base_pred: np.ndarray | None = None,
    stage2_pred: np.ndarray | None = None,
) -> np.ndarray:
    pieces = [X_flat]
    if base_pred is not None:
        pieces.append(base_pred.reshape(-1, 1))
    if stage2_pred is not None:
        pieces.append(stage2_pred.reshape(-1, 1))
    return np.hstack(pieces)


def ror_weighted_lgbm(
    ror_Xtr: np.ndarray,
    ror_ytr: np.ndarray,
    ror_Xva: np.ndarray,
    ror_Xte: np.ndarray,
    s2_va: np.ndarray,
    s2_te: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    ystd: float,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    model = lgb.LGBMRegressor(
        learning_rate=0.01,
        num_leaves=7,
        min_child_samples=50,
        subsample=0.5,
        colsample_bytree=0.4,
        reg_alpha=5.0,
        reg_lambda=5.0,
        n_estimators=200,
        verbose=-1,
        random_state=SEED,
    )
    model.fit(ror_Xtr, ror_ytr)
    va_pred = model.predict(ror_Xva)
    te_pred = model.predict(ror_Xte)

    best_lambda = 0.0
    best_va_rmse = rmse(y_val, s2_va)
    best_va = s2_va.copy()
    best_te = s2_te.copy()

    for lam in np.arange(0.0, 0.52, 0.02):
        cand_va = s2_va + lam * va_pred * ystd
        cand_te = s2_te + lam * te_pred * ystd
        cand_rmse = rmse(y_val, cand_va)
        if cand_rmse < best_va_rmse:
            best_lambda = float(lam)
            best_va_rmse = cand_rmse
            best_va = cand_va
            best_te = cand_te

    return best_va, best_te, best_va_rmse, f"WeightedLGBM(l={best_lambda:.2f})"


def ror_ridge(
    ror_Xtr: np.ndarray,
    ror_ytr: np.ndarray,
    ror_Xva: np.ndarray,
    ror_Xte: np.ndarray,
    s2_va: np.ndarray,
    s2_te: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    ystd: float,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    best_alpha = None
    best_lambda = None
    best_va_rmse = rmse(y_val, s2_va)
    best_va = s2_va.copy()
    best_te = s2_te.copy()

    for alpha in [0.1, 1.0, 10.0, 50.0, 100.0, 500.0]:
        model = Ridge(alpha=alpha)
        model.fit(ror_Xtr, ror_ytr)
        va_pred = model.predict(ror_Xva)
        te_pred = model.predict(ror_Xte)
        for lam in np.arange(0.0, 0.52, 0.02):
            cand_va = s2_va + lam * va_pred * ystd
            cand_te = s2_te + lam * te_pred * ystd
            cand_rmse = rmse(y_val, cand_va)
            if cand_rmse < best_va_rmse:
                best_alpha = alpha
                best_lambda = float(lam)
                best_va_rmse = cand_rmse
                best_va = cand_va
                best_te = cand_te

    desc = "Ridge(no-improve)"
    if best_alpha is not None and best_lambda is not None:
        desc = f"Ridge(a={best_alpha},l={best_lambda:.2f})"
    return best_va, best_te, best_va_rmse, desc


def ror_elasticnet(
    ror_Xtr: np.ndarray,
    ror_ytr: np.ndarray,
    ror_Xva: np.ndarray,
    ror_Xte: np.ndarray,
    s2_va: np.ndarray,
    s2_te: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    ystd: float,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    best_alpha = None
    best_l1 = None
    best_lambda = None
    best_va_rmse = rmse(y_val, s2_va)
    best_va = s2_va.copy()
    best_te = s2_te.copy()

    for alpha in [0.01, 0.1, 1.0, 10.0]:
        for l1_ratio in [0.1, 0.5, 0.9]:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, random_state=SEED)
            model.fit(ror_Xtr, ror_ytr)
            va_pred = model.predict(ror_Xva)
            te_pred = model.predict(ror_Xte)
            for lam in np.arange(0.0, 0.52, 0.02):
                cand_va = s2_va + lam * va_pred * ystd
                cand_te = s2_te + lam * te_pred * ystd
                cand_rmse = rmse(y_val, cand_va)
                if cand_rmse < best_va_rmse:
                    best_alpha = alpha
                    best_l1 = l1_ratio
                    best_lambda = float(lam)
                    best_va_rmse = cand_rmse
                    best_va = cand_va
                    best_te = cand_te

    desc = "ElasticNet(no-improve)"
    if best_alpha is not None and best_l1 is not None and best_lambda is not None:
        desc = f"ElasticNet(a={best_alpha},l1={best_l1},l={best_lambda:.2f})"
    return best_va, best_te, best_va_rmse, desc


def ror_oof_lgbm(
    ror_Xtr: np.ndarray,
    ror_ytr: np.ndarray,
    ror_Xva: np.ndarray,
    ror_Xte: np.ndarray,
    s2_va: np.ndarray,
    s2_te: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    ystd: float,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    n = len(ror_ytr)
    min_train = max(100, n // 3)
    fold_size = max(20, (n - min_train) // 5)
    models = []

    cursor = min_train
    while cursor < n:
        train_end = cursor
        val_end = min(n, cursor + fold_size)
        fold_model = lgb.LGBMRegressor(
            learning_rate=0.01,
            num_leaves=7,
            min_child_samples=50,
            subsample=0.5,
            colsample_bytree=0.4,
            reg_alpha=5.0,
            reg_lambda=5.0,
            n_estimators=150,
            verbose=-1,
            random_state=SEED,
        )
        fold_model.fit(ror_Xtr[:train_end], ror_ytr[:train_end])
        models.append(fold_model)
        cursor = val_end

    final_model = lgb.LGBMRegressor(
        learning_rate=0.01,
        num_leaves=7,
        min_child_samples=50,
        subsample=0.5,
        colsample_bytree=0.4,
        reg_alpha=5.0,
        reg_lambda=5.0,
        n_estimators=150,
        verbose=-1,
        random_state=SEED,
    )
    final_model.fit(ror_Xtr, ror_ytr)
    models.append(final_model)

    va_pred = np.mean([m.predict(ror_Xva) for m in models], axis=0)
    te_pred = np.mean([m.predict(ror_Xte) for m in models], axis=0)

    best_lambda = 0.0
    best_va_rmse = rmse(y_val, s2_va)
    best_va = s2_va.copy()
    best_te = s2_te.copy()
    for lam in np.arange(0.0, 0.52, 0.02):
        cand_va = s2_va + lam * va_pred * ystd
        cand_te = s2_te + lam * te_pred * ystd
        cand_rmse = rmse(y_val, cand_va)
        if cand_rmse < best_va_rmse:
            best_lambda = float(lam)
            best_va_rmse = cand_rmse
            best_va = cand_va
            best_te = cand_te
    return best_va, best_te, best_va_rmse, f"OOF_LGBM(l={best_lambda:.2f},folds={len(models)})"


def ror_augmented_lgbm(
    ror_Xtr: np.ndarray,
    ror_ytr: np.ndarray,
    ror_Xva: np.ndarray,
    ror_Xte: np.ndarray,
    s2_va: np.ndarray,
    s2_te: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    ystd: float,
    base_tr: np.ndarray,
    base_va: np.ndarray,
    base_te: np.ndarray,
    resid_tr: np.ndarray,
    resid_va: np.ndarray,
    resid_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    Xtr_aug = build_ror_augmented_features(ror_Xtr, base_pred=base_tr, stage2_pred=resid_tr)
    Xva_aug = build_ror_augmented_features(ror_Xva, base_pred=base_va, stage2_pred=resid_va)
    Xte_aug = build_ror_augmented_features(ror_Xte, base_pred=base_te, stage2_pred=resid_te)

    model = lgb.LGBMRegressor(
        learning_rate=0.01,
        num_leaves=7,
        min_child_samples=50,
        subsample=0.5,
        colsample_bytree=0.4,
        reg_alpha=5.0,
        reg_lambda=5.0,
        n_estimators=200,
        verbose=-1,
        random_state=SEED,
    )
    model.fit(Xtr_aug, ror_ytr)
    va_pred = model.predict(Xva_aug)
    te_pred = model.predict(Xte_aug)

    best_lambda = 0.0
    best_va_rmse = rmse(y_val, s2_va)
    best_va = s2_va.copy()
    best_te = s2_te.copy()
    for lam in np.arange(0.0, 0.52, 0.02):
        cand_va = s2_va + lam * va_pred * ystd
        cand_te = s2_te + lam * te_pred * ystd
        cand_rmse = rmse(y_val, cand_va)
        if cand_rmse < best_va_rmse:
            best_lambda = float(lam)
            best_va_rmse = cand_rmse
            best_va = cand_va
            best_te = cand_te
    return best_va, best_te, best_va_rmse, f"AugLGBM(l={best_lambda:.2f})"


def ror_ensemble(
    ror_Xtr: np.ndarray,
    ror_ytr: np.ndarray,
    ror_Xva: np.ndarray,
    ror_Xte: np.ndarray,
    s2_va: np.ndarray,
    s2_te: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    ystd: float,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    ridge = Ridge(alpha=10.0)
    ridge.fit(ror_Xtr, ror_ytr)
    ridge_va = ridge.predict(ror_Xva)
    ridge_te = ridge.predict(ror_Xte)

    lgbm = lgb.LGBMRegressor(
        learning_rate=0.01,
        num_leaves=7,
        min_child_samples=50,
        subsample=0.5,
        colsample_bytree=0.4,
        reg_alpha=5.0,
        reg_lambda=5.0,
        n_estimators=200,
        verbose=-1,
        random_state=SEED,
    )
    lgbm.fit(ror_Xtr, ror_ytr)
    lgb_va = lgbm.predict(ror_Xva)
    lgb_te = lgbm.predict(ror_Xte)

    best_mix = 0.0
    best_lambda = 0.0
    best_va_rmse = rmse(y_val, s2_va)
    best_va = s2_va.copy()
    best_te = s2_te.copy()
    for mix in np.arange(0.0, 1.01, 0.1):
        blend_va = mix * ridge_va + (1.0 - mix) * lgb_va
        blend_te = mix * ridge_te + (1.0 - mix) * lgb_te
        for lam in np.arange(0.0, 0.52, 0.02):
            cand_va = s2_va + lam * blend_va * ystd
            cand_te = s2_te + lam * blend_te * ystd
            cand_rmse = rmse(y_val, cand_va)
            if cand_rmse < best_va_rmse:
                best_mix = float(mix)
                best_lambda = float(lam)
                best_va_rmse = cand_rmse
                best_va = cand_va
                best_te = cand_te
    return best_va, best_te, best_va_rmse, f"Ensemble(m={best_mix:.2f},l={best_lambda:.2f})"


def ror_all_feature_lgbm(
    ror_Xtr_all: np.ndarray,
    ror_ytr: np.ndarray,
    ror_Xva_all: np.ndarray,
    ror_Xte_all: np.ndarray,
    s2_va: np.ndarray,
    s2_te: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    ystd: float,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    model = lgb.LGBMRegressor(
        learning_rate=0.01,
        num_leaves=7,
        min_child_samples=50,
        subsample=0.5,
        colsample_bytree=0.4,
        reg_alpha=10.0,
        reg_lambda=10.0,
        n_estimators=150,
        verbose=-1,
        random_state=SEED,
    )
    model.fit(ror_Xtr_all, ror_ytr)
    va_pred = model.predict(ror_Xva_all)
    te_pred = model.predict(ror_Xte_all)

    best_lambda = 0.0
    best_va_rmse = rmse(y_val, s2_va)
    best_va = s2_va.copy()
    best_te = s2_te.copy()
    for lam in np.arange(0.0, 0.52, 0.02):
        cand_va = s2_va + lam * va_pred * ystd
        cand_te = s2_te + lam * te_pred * ystd
        cand_rmse = rmse(y_val, cand_va)
        if cand_rmse < best_va_rmse:
            best_lambda = float(lam)
            best_va_rmse = cand_rmse
            best_va = cand_va
            best_te = cand_te
    return best_va, best_te, best_va_rmse, f"AllFeatLGBM(l={best_lambda:.2f})"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(DATA_PATH)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").set_index("dt")
    df.index.freq = "W-MON"

    bundle = prepare_sequences(df)
    screen_rows = []
    for base_name, resid_name in EXPERIMENTS:
        row, _ = run_stage2_experiment(
            bundle,
            base_name,
            resid_name,
            device,
            n_seeds=SCREEN_STAGE_SEEDS,
            epochs=SCREEN_EPOCHS,
            patience=SCREEN_PATIENCE,
            keep_artifacts=False,
        )
        screen_rows.append(row)
    screening_df = pd.DataFrame(screen_rows).sort_values("S2_Val_RMSE").reset_index(drop=True)
    screening_df.to_csv(f"{OUT_DIR}/stage2_screening.csv", index=False)

    confirm_pairs = list(screening_df.head(CONFIRM_TOPK)[["Base", "Residual"]].itertuples(index=False, name=None))
    confirm_rows = []
    artifacts: dict[str, dict[str, np.ndarray | str | float]] = {}
    for base_name, resid_name in confirm_pairs:
        row, artifact = run_stage2_experiment(
            bundle,
            base_name,
            resid_name,
            device,
            n_seeds=CONFIRM_STAGE_SEEDS,
            epochs=CONFIRM_EPOCHS,
            patience=CONFIRM_PATIENCE,
            keep_artifacts=True,
        )
        confirm_rows.append(row)
        artifacts[row["Experiment"]] = artifact

    stage2_df = pd.DataFrame(confirm_rows).sort_values("S2_Val_RMSE").reset_index(drop=True)
    stage2_df.to_csv(f"{OUT_DIR}/stage2_experiments.csv", index=False)

    top3 = stage2_df.head(3).copy()
    top3.to_csv(f"{OUT_DIR}/top3_by_validation.csv", index=False)

    ror_rows = []
    detail_rows = []
    y_val_arr = bundle.y_val.to_numpy(dtype=float)[-len(bundle.yva_seq) :]
    y_test_arr = bundle.y_test.to_numpy(dtype=float)[-len(bundle.yte_seq) :]

    for _, row in top3.iterrows():
        label = row["Experiment"]
        art = artifacts[label]

        ror_ytr = art["resid_train_target"] - art["resid_train"]
        ror_Xtr = bundle.Xtr_sc[SEQ_LEN:][: len(ror_ytr)]
        ror_Xva = bundle.Xva_sc[-len(bundle.yva_seq) :]
        ror_Xte = bundle.Xte_sc[-len(bundle.yte_seq) :]
        ror_Xtr_all = bundle.Xtr_all_sc[SEQ_LEN:][: len(ror_ytr)]
        ror_Xva_all = bundle.Xva_all_sc[-len(bundle.yva_seq) :]
        ror_Xte_all = bundle.Xte_all_sc[-len(bundle.yte_seq) :]

        usable = min(len(ror_ytr), len(ror_Xtr), len(ror_Xtr_all))
        ror_ytr = ror_ytr[:usable]
        ror_Xtr = ror_Xtr[:usable]
        ror_Xtr_all = ror_Xtr_all[:usable]

        base_train = art["base_train"][:usable]
        resid_train = art["resid_train"][:usable]
        base_val = art["base_val"][-len(bundle.yva_seq) :]
        base_test = art["base_test"][-len(bundle.yte_seq) :]
        resid_val = art["resid_val"][-len(bundle.yva_seq) :]
        resid_test = art["resid_test"][-len(bundle.yte_seq) :]
        s2_val = art["s2_val"]
        s2_test = art["s2_test"]

        strategies = {
            "A": ror_weighted_lgbm(
                ror_Xtr,
                ror_ytr,
                ror_Xva,
                ror_Xte,
                s2_val,
                s2_test,
                y_val_arr,
                y_test_arr,
                bundle.ystd,
            ),
            "B": ror_ridge(
                ror_Xtr,
                ror_ytr,
                ror_Xva,
                ror_Xte,
                s2_val,
                s2_test,
                y_val_arr,
                y_test_arr,
                bundle.ystd,
            ),
            "C": ror_elasticnet(
                ror_Xtr,
                ror_ytr,
                ror_Xva,
                ror_Xte,
                s2_val,
                s2_test,
                y_val_arr,
                y_test_arr,
                bundle.ystd,
            ),
            "D": ror_oof_lgbm(
                ror_Xtr,
                ror_ytr,
                ror_Xva,
                ror_Xte,
                s2_val,
                s2_test,
                y_val_arr,
                y_test_arr,
                bundle.ystd,
            ),
            "E": ror_augmented_lgbm(
                ror_Xtr,
                ror_ytr,
                ror_Xva,
                ror_Xte,
                s2_val,
                s2_test,
                y_val_arr,
                y_test_arr,
                bundle.ystd,
                base_tr=base_train,
                base_va=base_val,
                base_te=base_test,
                resid_tr=resid_train,
                resid_va=resid_val,
                resid_te=resid_test,
            ),
            "F": ror_ensemble(
                ror_Xtr,
                ror_ytr,
                ror_Xva,
                ror_Xte,
                s2_val,
                s2_test,
                y_val_arr,
                y_test_arr,
                bundle.ystd,
            ),
            "G": ror_all_feature_lgbm(
                ror_Xtr_all,
                ror_ytr,
                ror_Xva_all,
                ror_Xte_all,
                s2_val,
                s2_test,
                y_val_arr,
                y_test_arr,
                bundle.ystd,
            ),
        }

        best_key = None
        best_tuple = None
        best_val_rmse = row["S2_Val_RMSE"]
        final_test_rmse = row["S2_Test_RMSE"]
        for key, strategy_out in strategies.items():
            pred_val, pred_test, val_rmse, desc = strategy_out
            test_rmse = rmse(y_test_arr, pred_test)
            detail_rows.append(
                {
                    "Experiment": label,
                    "Strategy": key,
                    "Description": desc,
                    "Val_RMSE": val_rmse,
                    "Test_RMSE": test_rmse,
                    "Improves_S2_Validation": val_rmse < row["S2_Val_RMSE"],
                }
            )
            if val_rmse < best_val_rmse:
                best_key = key
                best_tuple = (pred_val, pred_test, val_rmse, desc)
                best_val_rmse = val_rmse
                final_test_rmse = test_rmse

        if best_key is None or best_tuple is None:
            ror_rows.append(
                {
                    "Experiment": label,
                    "Base": row["Base"],
                    "Residual": row["Residual"],
                    "S2_Val_RMSE": row["S2_Val_RMSE"],
                    "S2_Test_RMSE": row["S2_Test_RMSE"],
                    "S2_Test_MAE": row["S2_Test_MAE"],
                    "S2_Test_MAPE(%)": row["S2_Test_MAPE(%)"],
                    "S2_Val_NRMSE(%)": row["S2_Val_NRMSE(%)"],
                    "S2_Val_R2": row["S2_Val_R2"],
                    "S2_Test_NRMSE(%)": row["S2_Test_NRMSE(%)"],
                    "S2_Test_R2": row["S2_Test_R2"],
                    "RoR_Strategy": "None",
                    "RoR_Description": "No validation improvement",
                    "Final_Val_RMSE": row["S2_Val_RMSE"],
                    "Final_Test_RMSE": row["S2_Test_RMSE"],
                    "Final_Test_MAE": row["S2_Test_MAE"],
                    "Final_Test_MAPE(%)": row["S2_Test_MAPE(%)"],
                    "Final_Val_NRMSE(%)": row["S2_Val_NRMSE(%)"],
                    "Final_Val_R2": row["S2_Val_R2"],
                    "Final_Test_NRMSE(%)": row["S2_Test_NRMSE(%)"],
                    "Final_Test_R2": row["S2_Test_R2"],
                }
            )
        else:
            _, pred_test, val_rmse, desc = best_tuple
            pred_val, _, _, _ = best_tuple
            ror_rows.append(
                {
                    "Experiment": label,
                    "Base": row["Base"],
                    "Residual": row["Residual"],
                    "S2_Val_RMSE": row["S2_Val_RMSE"],
                    "S2_Test_RMSE": row["S2_Test_RMSE"],
                    "S2_Test_MAE": row["S2_Test_MAE"],
                    "S2_Test_MAPE(%)": row["S2_Test_MAPE(%)"],
                    "S2_Val_NRMSE(%)": row["S2_Val_NRMSE(%)"],
                    "S2_Val_R2": row["S2_Val_R2"],
                    "S2_Test_NRMSE(%)": row["S2_Test_NRMSE(%)"],
                    "S2_Test_R2": row["S2_Test_R2"],
                    "RoR_Strategy": best_key,
                    "RoR_Description": desc,
                    "Final_Val_RMSE": val_rmse,
                    "Final_Test_RMSE": final_test_rmse,
                    "Final_Test_MAE": float(mean_absolute_error(y_test_arr, pred_test)),
                    "Final_Test_MAPE(%)": mape(y_test_arr, pred_test),
                    "Final_Val_NRMSE(%)": nrmse_pct(y_val_arr, pred_val),
                    "Final_Val_R2": r2_metric(y_val_arr, pred_val),
                    "Final_Test_NRMSE(%)": nrmse_pct(y_test_arr, pred_test),
                    "Final_Test_R2": r2_metric(y_test_arr, pred_test),
                }
            )

    ror_df = pd.DataFrame(ror_rows).sort_values("Final_Val_RMSE").reset_index(drop=True)
    ror_detail_df = pd.DataFrame(detail_rows).sort_values(["Experiment", "Strategy"]).reset_index(drop=True)
    ror_df.to_csv(f"{OUT_DIR}/top3_ror_results.csv", index=False)
    ror_detail_df.to_csv(f"{OUT_DIR}/ror_strategy_details.csv", index=False)

    config = {
        "target": TARGET,
        "train_size": int(len(bundle.y_train)),
        "val_size": int(len(bundle.y_val)),
        "test_size": int(len(bundle.y_test)),
        "feature_selection": {
            "method": "train-only SHAP + TimeSeriesCV",
            "selected_count": len(bundle.selected_features),
            "selected_features": bundle.selected_features,
            "total_features": len(bundle.all_features),
        },
        "experiments": EXPERIMENTS,
        "screening": {
            "pairs_tested": len(EXPERIMENTS),
            "stage_seeds": SCREEN_STAGE_SEEDS,
            "epochs": SCREEN_EPOCHS,
            "patience": SCREEN_PATIENCE,
        },
        "confirmatory": {
            "topk_from_screening": CONFIRM_TOPK,
            "stage_seeds": CONFIRM_STAGE_SEEDS,
            "epochs": CONFIRM_EPOCHS,
            "patience": CONFIRM_PATIENCE,
        },
        "top3_selection_rule": "Lowest confirmatory S2 validation RMSE after 25-pair screening",
        "ror_strategies": {
            "A": "Weighted LightGBM",
            "B": "Ridge",
            "C": "ElasticNet",
            "D": "OOF LightGBM",
            "E": "Augmented LightGBM",
            "F": "Ensemble Ridge+LightGBM",
            "G": "All-feature LightGBM",
        },
    }
    with open(f"{OUT_DIR}/config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Oil Transformer Advanced Clean Pipeline")
    print("=" * 72)
    print(f"Selected features : {len(bundle.selected_features)} / {len(bundle.all_features)}")
    print("Top-5 screening candidates:")
    print(screening_df[["Experiment", "S2_Val_RMSE", "S2_Test_RMSE"]].head(5).to_string(index=False))
    print()
    print("Top-3 by validation:")
    print(top3[["Experiment", "S2_Val_RMSE", "S2_Test_RMSE"]].to_string(index=False))
    print()
    print("Top-3 after pure RoR validation gating:")
    print(ror_df.to_string(index=False))
    print()
    print(f"Output directory  : {OUT_DIR}")


if __name__ == "__main__":
    main()
