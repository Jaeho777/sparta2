from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from torch.utils.data import DataLoader, Dataset

import oil_transformer_advanced_clean as tfm


DATA_PATH = "data_weekly_260120.csv"
TARGET = "Com_BrentCrudeOil"
OUT_DIR = "output_oil_common_protocol"
TEST_STARTS = [
    "2024-11-25",
    "2025-02-17",
    "2025-05-12",
    "2025-08-04",
    "2025-10-27",
]
WINDOW = 12

TRANSFORMER_SEEDS = 2
TRANSFORMER_EPOCHS = 80
TRANSFORMER_PATIENCE = 15

ELASTIC_ALPHA = 0.01
ELASTIC_L1 = 0.1
ELASTIC_LAMBDA = 0.50

STL_SEQ_LEN = 24
STL_PRED_LEN = 1
STL_HIDDEN = 64
STL_EPOCHS = 160
STL_PATIENCE = 25
STL_BATCH_SIZE = 32
STL_SEEDS = 3
STL_TOPK = 2
STL_LR = 1e-3


@dataclass
class Window:
    label: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


class NLinearWithExog(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, n_exog: int, d_hidden: int = 64):
        super().__init__()
        self.linear_residual = nn.Linear(seq_len, d_hidden)
        self.linear_exog = nn.Linear(n_exog, d_hidden)
        self.linear_out = nn.Linear(d_hidden * 2, pred_len)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x_seq: torch.Tensor, x_exog: torch.Tensor) -> torch.Tensor:
        last_val = x_seq[:, -1:].detach()
        x_norm = x_seq - last_val
        h_res = torch.relu(self.linear_residual(x_norm))
        h_exog = torch.relu(self.linear_exog(x_exog))
        h = self.dropout(torch.cat([h_res, h_exog], dim=-1))
        return self.linear_out(h) + last_val


class ResidualDataset(Dataset):
    def __init__(self, resid_values: np.ndarray, exog_values: np.ndarray, seq_len: int, pred_len: int):
        self.resid = resid_values.astype(np.float32)
        self.exog = exog_values.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_samples = len(resid_values) - seq_len - pred_len + 1

    def __len__(self) -> int:
        return max(0, self.n_samples)

    def __getitem__(self, idx: int):
        s = idx
        e = s + self.seq_len
        return (
            torch.tensor(self.resid[s:e]),
            torch.tensor(self.exog[e - 1]),
            torch.tensor(self.resid[e : e + self.pred_len]),
        )


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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").set_index("dt")
    df.index.freq = "W-MON"
    return df


def build_windows(df: pd.DataFrame) -> list[Window]:
    windows = []
    for test_start in TEST_STARTS:
        test_loc = df.index.get_loc(pd.Timestamp(test_start))
        val_loc = test_loc - WINDOW
        if val_loc <= 0:
            raise ValueError(f"Not enough history for {test_start}")
        val_start = df.index[val_loc]
        train_end = df.index[val_loc - 1]
        test_end = df.index[test_loc + WINDOW - 1]
        windows.append(
            Window(
                label=test_start,
                train_end=str(train_end.date()),
                val_start=str(val_start.date()),
                val_end=str(df.index[test_loc - 1].date()),
                test_start=test_start,
                test_end=str(test_end.date()),
            )
        )
    return windows


def calc_random_walk(y: pd.Series, indices: pd.Index) -> np.ndarray:
    return y.shift(1).loc[indices].to_numpy(dtype=float)


def calc_two_point_linear(y: pd.Series, indices: pd.Index) -> np.ndarray:
    preds = []
    for idx in indices:
        loc = y.index.get_loc(idx)
        prev_1 = float(y.iloc[loc - 1])
        prev_2 = float(y.iloc[loc - 2])
        preds.append(prev_1 + (prev_1 - prev_2))
    return np.asarray(preds, dtype=float)


def run_baseline_window(df: pd.DataFrame, window: Window) -> list[dict[str, object]]:
    mask_train = df.index < window.val_start
    mask_val = (df.index >= window.val_start) & (df.index < window.test_start)
    mask_test = (df.index >= window.test_start) & (df.index <= window.test_end)

    feature_cols = [c for c in df.columns if c != TARGET]
    X_all = df[feature_cols].shift(1)
    y_all = df[TARGET]

    X_train = X_all.loc[mask_train]
    X_val = X_all.loc[mask_val]
    X_test = X_all.loc[mask_test]
    y_train = y_all.loc[mask_train]
    y_val = y_all.loc[mask_val]
    y_test = y_all.loc[mask_test]

    two_point_val = calc_two_point_linear(y_all, y_val.index)
    two_point_test = calc_two_point_linear(y_all, y_test.index)
    rw_val = calc_random_walk(y_all, y_val.index)
    rw_test = calc_random_walk(y_all, y_test.index)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    gb = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    gb.fit(X_train_imp, y_train)
    gb_val = gb.predict(X_val_imp)
    gb_test = gb.predict(X_test_imp)

    best_weight = None
    best_val_rmse = float("inf")
    best_test_rmse = None
    best_val_pred = None
    best_test_pred = None
    for weight in [0.70, 0.80, 0.90]:
        val_pred = weight * two_point_val + (1.0 - weight) * gb_val
        test_pred = weight * two_point_test + (1.0 - weight) * gb_test
        val_score = rmse(y_val.to_numpy(dtype=float), val_pred)
        test_score = rmse(y_test.to_numpy(dtype=float), test_pred)
        if val_score < best_val_rmse:
            best_weight = weight
            best_val_rmse = val_score
            best_test_rmse = test_score
            best_val_pred = val_pred
            best_test_pred = test_pred

    if best_val_pred is None or best_test_pred is None:
        raise RuntimeError("Failed to choose a baseline hybrid weight.")

    return [
        {
            "window": window.label,
            "family": "benchmark",
            "model": "RandomWalk_1step",
            "val_rmse": rmse(y_val.to_numpy(dtype=float), rw_val),
            "val_mae": float(mean_absolute_error(y_val.to_numpy(dtype=float), rw_val)),
            "val_mape": mape(y_val.to_numpy(dtype=float), rw_val),
            "val_nrmse": nrmse_pct(y_val.to_numpy(dtype=float), rw_val),
            "val_r2": r2_metric(y_val.to_numpy(dtype=float), rw_val),
            "test_rmse": rmse(y_test.to_numpy(dtype=float), rw_test),
            "test_mae": float(mean_absolute_error(y_test.to_numpy(dtype=float), rw_test)),
            "test_mape": mape(y_test.to_numpy(dtype=float), rw_test),
            "test_nrmse": nrmse_pct(y_test.to_numpy(dtype=float), rw_test),
            "test_r2": r2_metric(y_test.to_numpy(dtype=float), rw_test),
            "extra": "y_hat_t = y_(t-1)",
        },
        {
            "window": window.label,
            "family": "baseline",
            "model": "Hybrid_TwoPointLinear_GB",
            "val_rmse": best_val_rmse,
            "test_rmse": best_test_rmse,
            "val_mae": float(mean_absolute_error(y_val.to_numpy(dtype=float), best_val_pred)),
            "val_mape": mape(y_val.to_numpy(dtype=float), best_val_pred),
            "val_nrmse": nrmse_pct(y_val.to_numpy(dtype=float), best_val_pred),
            "val_r2": r2_metric(y_val.to_numpy(dtype=float), best_val_pred),
            "test_mae": float(mean_absolute_error(y_test.to_numpy(dtype=float), best_test_pred)),
            "test_mape": mape(y_test.to_numpy(dtype=float), best_test_pred),
            "test_nrmse": nrmse_pct(y_test.to_numpy(dtype=float), best_test_pred),
            "test_r2": r2_metric(y_test.to_numpy(dtype=float), best_test_pred),
            "extra": f"validation-selected weight={best_weight:.2f}",
        },
    ]


def run_transformer_stage2(
    df: pd.DataFrame,
    window: Window,
    base_name: str,
    residual_name: str,
    device: torch.device,
    origin_dir: str,
) -> tuple[dict[str, object], tfm.SequenceBundle, dict[str, np.ndarray | str | float]]:
    old_val, old_test, old_out = tfm.VAL_START, tfm.TEST_START, tfm.OUT_DIR
    try:
        tfm.VAL_START = window.val_start
        tfm.TEST_START = window.test_start
        tfm.OUT_DIR = origin_dir
        bundle = tfm.prepare_sequences(df)
        row, artifacts = tfm.run_stage2_experiment(
            bundle,
            base_name,
            residual_name,
            device,
            n_seeds=TRANSFORMER_SEEDS,
            epochs=TRANSFORMER_EPOCHS,
            patience=TRANSFORMER_PATIENCE,
            keep_artifacts=True,
        )
    finally:
        tfm.VAL_START = old_val
        tfm.TEST_START = old_test
        tfm.OUT_DIR = old_out

    out = {
        "window": window.label,
        "family": "transformer",
        "model": f"{base_name}+{residual_name}",
        "val_rmse": float(row["S2_Val_RMSE"]),
        "val_mae": np.nan,
        "val_mape": np.nan,
        "val_nrmse": float(row["S2_Val_NRMSE(%)"]),
        "val_r2": float(row["S2_Val_R2"]),
        "test_rmse": float(row["S2_Test_RMSE"]),
        "test_mae": float(row["S2_Test_MAE"]),
        "test_mape": float(row["S2_Test_MAPE(%)"]),
        "test_nrmse": float(row["S2_Test_NRMSE(%)"]),
        "test_r2": float(row["S2_Test_R2"]),
        "extra": (
            f"features={len(bundle.selected_features)} "
            f"seeds={TRANSFORMER_SEEDS} epochs={TRANSFORMER_EPOCHS}"
        ),
    }
    return out, bundle, artifacts


def run_transformer_stage3(
    window: Window,
    bundle: tfm.SequenceBundle,
    artifacts: dict[str, np.ndarray | str | float],
) -> dict[str, object]:
    y_val = bundle.y_val.to_numpy(dtype=float)[-len(bundle.yva_seq) :]
    y_test = bundle.y_test.to_numpy(dtype=float)[-len(bundle.yte_seq) :]

    ror_ytr = artifacts["resid_train_target"] - artifacts["resid_train"]
    ror_Xtr = bundle.Xtr_sc[tfm.SEQ_LEN:][: len(ror_ytr)]
    ror_Xva = bundle.Xva_sc[-len(bundle.yva_seq) :]
    ror_Xte = bundle.Xte_sc[-len(bundle.yte_seq) :]

    usable = min(len(ror_ytr), len(ror_Xtr))
    ror_ytr = ror_ytr[:usable]
    ror_Xtr = ror_Xtr[:usable]

    model = ElasticNet(
        alpha=ELASTIC_ALPHA,
        l1_ratio=ELASTIC_L1,
        max_iter=20000,
        random_state=tfm.SEED,
    )
    model.fit(ror_Xtr, ror_ytr)
    ror_val = model.predict(ror_Xva)
    ror_test = model.predict(ror_Xte)

    s2_val = artifacts["s2_val"]
    s2_test = artifacts["s2_test"]
    gated = rmse(y_val, s2_val + ELASTIC_LAMBDA * ror_val * bundle.ystd) < rmse(y_val, s2_val)
    lam = ELASTIC_LAMBDA if gated else 0.0
    final_val = s2_val + lam * ror_val * bundle.ystd
    final_test = s2_test + lam * ror_test * bundle.ystd
    return {
        "window": window.label,
        "family": "transformer",
        "model": "PatchTST+Transformer+ElasticNetRoR",
        "val_rmse": rmse(y_val, final_val),
        "test_rmse": rmse(y_test, final_test),
        "val_mae": float(mean_absolute_error(y_val, final_val)),
        "val_mape": mape(y_val, final_val),
        "val_nrmse": nrmse_pct(y_val, final_val),
        "val_r2": r2_metric(y_val, final_val),
        "test_mae": float(mean_absolute_error(y_test, final_test)),
        "test_mape": mape(y_test, final_test),
        "test_nrmse": nrmse_pct(y_test, final_test),
        "test_r2": r2_metric(y_test, final_test),
        "extra": (
            f"alpha={ELASTIC_ALPHA} l1={ELASTIC_L1} "
            f"lambda={lam:.2f} gate={'on' if gated else 'off'}"
        ),
    }


def build_stl_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_groups = {
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
    base_features = []
    for cols in feature_groups.values():
        for col in cols:
            if col in df.columns:
                base_features.append(col)

    derived = pd.DataFrame(index=df.index)
    for col in base_features:
        derived[f"{col}_ret"] = df[col].pct_change()

    derived["Spread_US_10Y_2Y"] = df["Bonds_US_10Y"] - df["Bonds_US_2Y"]
    derived["Spread_Crack"] = df["Com_Gasoline"] - df[TARGET]
    derived["Ratio_Gold_Oil"] = df["Com_Gold"] / df[TARGET]
    for col in ["Com_Gasoline", "Com_NaturalGas", "Idx_SnPVIX", "Idx_DxyUSD"]:
        ma4 = df[col].rolling(4).mean()
        ma12 = df[col].rolling(12).mean()
        derived[f"{col}_ma4_ratio"] = df[col] / ma4 - 1.0
        derived[f"{col}_ma12_ratio"] = df[col] / ma12 - 1.0

    return pd.concat([df[base_features], derived], axis=1).shift(1)


def es_stepwise_forecast(train_series: pd.Series, eval_series: pd.Series) -> np.ndarray:
    history = train_series.copy()
    preds = []
    for idx, actual in eval_series.items():
        fit = ExponentialSmoothing(
            history,
            trend="add",
            seasonal="add",
            seasonal_periods=52,
            initialization_method="estimated",
            use_boxcox=False,
        ).fit(optimized=True)
        pred = float(fit.forecast(1).iloc[0])
        preds.append(pred)
        history = pd.concat([history, pd.Series([actual], index=[idx])])
    return np.asarray(preds, dtype=float)


def predict_nlinear(models: list[nn.Module], resid_buffer_init: np.ndarray, X_scaled: np.ndarray, actual_resids: np.ndarray, device: torch.device) -> np.ndarray:
    all_preds = []
    for model in models:
        model.eval()
        preds = []
        buffer = list(resid_buffer_init.copy())
        with torch.no_grad():
            for i in range(len(X_scaled)):
                x_seq = torch.tensor(np.asarray(buffer[-STL_SEQ_LEN:], dtype=np.float32)).unsqueeze(0).to(device)
                x_exog = torch.tensor(X_scaled[i : i + 1].astype(np.float32)).to(device)
                pred = model(x_seq, x_exog).cpu().numpy().flatten()[0]
                preds.append(pred)
                buffer.append(float(actual_resids[i]))
        all_preds.append(preds)
    return np.mean(all_preds, axis=0)


def train_stl_nlinear(
    resid_train: np.ndarray,
    resid_val: np.ndarray,
    X_train_scaled: np.ndarray,
    X_val_scaled: np.ndarray,
    device: torch.device,
) -> list[nn.Module]:
    n_exog = X_train_scaled.shape[1]
    train_dataset = ResidualDataset(resid_train, X_train_scaled, STL_SEQ_LEN, STL_PRED_LEN)
    train_loader = DataLoader(train_dataset, batch_size=STL_BATCH_SIZE, shuffle=True)
    ranked_models = []

    for seed_offset in range(STL_SEEDS):
        set_seed(42 + seed_offset)
        model = NLinearWithExog(STL_SEQ_LEN, STL_PRED_LEN, n_exog, STL_HIDDEN).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=STL_LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STL_EPOCHS)
        criterion = nn.MSELoss()

        best_score = float("inf")
        best_state = None
        wait = 0

        for _ in range(STL_EPOCHS):
            model.train()
            for x_seq, x_exog, y_true in train_loader:
                x_seq = x_seq.to(device)
                x_exog = x_exog.to(device)
                y_true = y_true.to(device)
                pred = model(x_seq, x_exog)
                loss = criterion(pred, y_true)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            val_preds = []
            buffer = list(resid_train[-STL_SEQ_LEN:])
            with torch.no_grad():
                for i in range(len(resid_val)):
                    x_seq = torch.tensor(np.asarray(buffer[-STL_SEQ_LEN:], dtype=np.float32)).unsqueeze(0).to(device)
                    x_exog = torch.tensor(X_val_scaled[i : i + 1].astype(np.float32)).to(device)
                    pred = model(x_seq, x_exog).cpu().numpy().flatten()[0]
                    val_preds.append(pred)
                    buffer.append(float(resid_val[i]))
            score = rmse(resid_val, np.asarray(val_preds, dtype=float))
            if score < best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= STL_PATIENCE:
                    break

        if best_state is None:
            raise RuntimeError("Failed to train STL NLinear model.")
        model.load_state_dict(best_state)
        ranked_models.append((model, best_score))

    ranked_models.sort(key=lambda item: item[1])
    return [model for model, _ in ranked_models[:STL_TOPK]]


def run_stl_window(df: pd.DataFrame, window: Window, device: torch.device) -> list[dict[str, object]]:
    mask_train = df.index < window.val_start
    mask_val = (df.index >= window.val_start) & (df.index < window.test_start)
    mask_test = (df.index >= window.test_start) & (df.index <= window.test_end)

    y_train = df.loc[mask_train, TARGET]
    y_val = df.loc[mask_val, TARGET]
    y_test = df.loc[mask_test, TARGET]

    X_lagged = build_stl_features(df)
    X_train = np.nan_to_num(X_lagged.loc[mask_train].to_numpy(), nan=0.0)
    X_val = np.nan_to_num(X_lagged.loc[mask_val].to_numpy(), nan=0.0)
    X_test = np.nan_to_num(X_lagged.loc[mask_test].to_numpy(), nan=0.0)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    es_fit_train = ExponentialSmoothing(
        y_train,
        trend="add",
        seasonal="add",
        seasonal_periods=52,
        initialization_method="estimated",
        use_boxcox=False,
    ).fit(optimized=True)
    baseline_train = es_fit_train.fittedvalues.to_numpy(dtype=float)
    baseline_val = es_stepwise_forecast(y_train, y_val)
    baseline_test = es_stepwise_forecast(pd.concat([y_train, y_val]), y_test)

    resid_train = np.nan_to_num(y_train.to_numpy(dtype=float) - baseline_train, nan=0.0).astype(np.float32)
    resid_val = (y_val.to_numpy(dtype=float) - baseline_val).astype(np.float32)
    resid_test = (y_test.to_numpy(dtype=float) - baseline_test).astype(np.float32)

    top_models = train_stl_nlinear(resid_train, resid_val, X_train_scaled, X_val_scaled, device)
    nlinear_train_all = []
    for model in top_models:
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(STL_SEQ_LEN, len(resid_train)):
                x_seq = torch.tensor(resid_train[i - STL_SEQ_LEN : i]).unsqueeze(0).to(device)
                x_exog = torch.tensor(X_train_scaled[i : i + 1].astype(np.float32)).to(device)
                pred = model(x_seq, x_exog).cpu().numpy().flatten()[0]
                preds.append(pred)
        nlinear_train_all.append(preds)
    nlinear_train = np.mean(nlinear_train_all, axis=0)

    nlinear_val = predict_nlinear(top_models, resid_train[-STL_SEQ_LEN:], X_val_scaled, resid_val, device)
    nlinear_test = predict_nlinear(
        top_models,
        np.concatenate([resid_train, resid_val])[-STL_SEQ_LEN:],
        X_test_scaled,
        resid_test,
        device,
    )

    stage2_val = baseline_val + nlinear_val
    stage2_test = baseline_test + nlinear_test

    ror_train = resid_train[STL_SEQ_LEN:] - nlinear_train
    ror_X_train = X_train_scaled[STL_SEQ_LEN:]
    ror_lgb = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        learning_rate=0.02,
        num_leaves=10,
        min_child_samples=40,
        subsample=0.6,
        colsample_bytree=0.5,
        reg_alpha=2.0,
        reg_lambda=2.0,
        n_estimators=300,
        verbose=-1,
        random_state=42,
    )
    ror_lgb.fit(ror_X_train, ror_train)
    ror_val = ror_lgb.predict(X_val_scaled)
    ror_test = ror_lgb.predict(X_test_scaled)

    stage3_val_candidate = stage2_val + ror_val
    gate_on = rmse(y_val.to_numpy(dtype=float), stage3_val_candidate) < rmse(y_val.to_numpy(dtype=float), stage2_val)
    lam = 1.0 if gate_on else 0.0
    stage3_val = stage2_val + lam * ror_val
    stage3_test = stage2_test + lam * ror_test

    return [
        {
            "window": window.label,
            "family": "stl",
            "model": "ExpSmoothing",
            "val_rmse": rmse(y_val.to_numpy(dtype=float), baseline_val),
            "val_mae": float(mean_absolute_error(y_val.to_numpy(dtype=float), baseline_val)),
            "val_mape": mape(y_val.to_numpy(dtype=float), baseline_val),
            "val_nrmse": nrmse_pct(y_val.to_numpy(dtype=float), baseline_val),
            "val_r2": r2_metric(y_val.to_numpy(dtype=float), baseline_val),
            "test_rmse": rmse(y_test.to_numpy(dtype=float), baseline_test),
            "test_mae": float(mean_absolute_error(y_test.to_numpy(dtype=float), baseline_test)),
            "test_mape": mape(y_test.to_numpy(dtype=float), baseline_test),
            "test_nrmse": nrmse_pct(y_test.to_numpy(dtype=float), baseline_test),
            "test_r2": r2_metric(y_test.to_numpy(dtype=float), baseline_test),
            "extra": "one-step expanding refit",
        },
        {
            "window": window.label,
            "family": "stl",
            "model": "ExpSmoothing+NLinear",
            "val_rmse": rmse(y_val.to_numpy(dtype=float), stage2_val),
            "val_mae": float(mean_absolute_error(y_val.to_numpy(dtype=float), stage2_val)),
            "val_mape": mape(y_val.to_numpy(dtype=float), stage2_val),
            "val_nrmse": nrmse_pct(y_val.to_numpy(dtype=float), stage2_val),
            "val_r2": r2_metric(y_val.to_numpy(dtype=float), stage2_val),
            "test_rmse": rmse(y_test.to_numpy(dtype=float), stage2_test),
            "test_mae": float(mean_absolute_error(y_test.to_numpy(dtype=float), stage2_test)),
            "test_mape": mape(y_test.to_numpy(dtype=float), stage2_test),
            "test_nrmse": nrmse_pct(y_test.to_numpy(dtype=float), stage2_test),
            "test_r2": r2_metric(y_test.to_numpy(dtype=float), stage2_test),
            "extra": f"topk={STL_TOPK} seeds={STL_SEEDS}",
        },
        {
            "window": window.label,
            "family": "stl",
            "model": "ExpSmoothing+NLinear+LightGBM",
            "val_rmse": rmse(y_val.to_numpy(dtype=float), stage3_val),
            "val_mae": float(mean_absolute_error(y_val.to_numpy(dtype=float), stage3_val)),
            "val_mape": mape(y_val.to_numpy(dtype=float), stage3_val),
            "val_nrmse": nrmse_pct(y_val.to_numpy(dtype=float), stage3_val),
            "val_r2": r2_metric(y_val.to_numpy(dtype=float), stage3_val),
            "test_rmse": rmse(y_test.to_numpy(dtype=float), stage3_test),
            "test_mae": float(mean_absolute_error(y_test.to_numpy(dtype=float), stage3_test)),
            "test_mape": mape(y_test.to_numpy(dtype=float), stage3_test),
            "test_nrmse": nrmse_pct(y_test.to_numpy(dtype=float), stage3_test),
            "test_r2": r2_metric(y_test.to_numpy(dtype=float), stage3_test),
            "extra": f"gate={'on' if gate_on else 'off'}",
        },
    ]


def summarize(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df.groupby(["family", "model"], as_index=False)
        .agg(
            mean_val_rmse=("val_rmse", "mean"),
            std_val_rmse=("val_rmse", "std"),
            mean_val_mae=("val_mae", "mean"),
            std_val_mae=("val_mae", "std"),
            mean_val_mape=("val_mape", "mean"),
            std_val_mape=("val_mape", "std"),
            mean_val_nrmse=("val_nrmse", "mean"),
            std_val_nrmse=("val_nrmse", "std"),
            mean_val_r2=("val_r2", "mean"),
            std_val_r2=("val_r2", "std"),
            mean_test_rmse=("test_rmse", "mean"),
            std_test_rmse=("test_rmse", "std"),
            mean_test_mae=("test_mae", "mean"),
            std_test_mae=("test_mae", "std"),
            mean_test_mape=("test_mape", "mean"),
            std_test_mape=("test_mape", "std"),
            mean_test_nrmse=("test_nrmse", "mean"),
            std_test_nrmse=("test_nrmse", "std"),
            mean_test_r2=("test_r2", "mean"),
            std_test_r2=("test_r2", "std"),
            n_windows=("window", "count"),
        )
        .sort_values(["mean_test_rmse", "mean_val_rmse"])
        .reset_index(drop=True)
    )
    summary["mean_val_rmse"] = summary["mean_val_rmse"].round(4)
    summary["std_val_rmse"] = summary["std_val_rmse"].fillna(0.0).round(4)
    summary["mean_val_mae"] = summary["mean_val_mae"].round(4)
    summary["std_val_mae"] = summary["std_val_mae"].fillna(0.0).round(4)
    summary["mean_val_mape"] = summary["mean_val_mape"].round(4)
    summary["std_val_mape"] = summary["std_val_mape"].fillna(0.0).round(4)
    summary["mean_val_nrmse"] = summary["mean_val_nrmse"].round(4)
    summary["std_val_nrmse"] = summary["std_val_nrmse"].fillna(0.0).round(4)
    summary["mean_val_r2"] = summary["mean_val_r2"].round(4)
    summary["std_val_r2"] = summary["std_val_r2"].fillna(0.0).round(4)
    summary["mean_test_rmse"] = summary["mean_test_rmse"].round(4)
    summary["std_test_rmse"] = summary["std_test_rmse"].fillna(0.0).round(4)
    summary["mean_test_mae"] = summary["mean_test_mae"].round(4)
    summary["std_test_mae"] = summary["std_test_mae"].fillna(0.0).round(4)
    summary["mean_test_mape"] = summary["mean_test_mape"].round(4)
    summary["std_test_mape"] = summary["std_test_mape"].fillna(0.0).round(4)
    summary["mean_test_nrmse"] = summary["mean_test_nrmse"].round(4)
    summary["std_test_nrmse"] = summary["std_test_nrmse"].fillna(0.0).round(4)
    summary["mean_test_r2"] = summary["mean_test_r2"].round(4)
    summary["std_test_r2"] = summary["std_test_r2"].fillna(0.0).round(4)
    return summary


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_frame()
    windows = build_windows(df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict[str, object]] = []
    for window in windows:
        rows.extend(run_baseline_window(df, window))

        origin_dir = os.path.join(OUT_DIR, window.label)
        os.makedirs(origin_dir, exist_ok=True)

        s2_pt_tf, bundle_pt_tf, art_pt_tf = run_transformer_stage2(
            df,
            window,
            base_name="PatchTST",
            residual_name="Transformer",
            device=device,
            origin_dir=origin_dir,
        )
        rows.append(s2_pt_tf)
        rows.append(run_transformer_stage3(window, bundle_pt_tf, art_pt_tf))

        s2_pt_it, _, _ = run_transformer_stage2(
            df,
            window,
            base_name="PatchTST",
            residual_name="iTransformer",
            device=device,
            origin_dir=origin_dir,
        )
        rows.append(s2_pt_it)

        rows.extend(run_stl_window(df, window, device))

    results_df = pd.DataFrame(rows)
    summary_df = summarize(results_df)

    results_df.to_csv(f"{OUT_DIR}/rolling_origin_results.csv", index=False)
    summary_df.to_csv(f"{OUT_DIR}/rolling_origin_summary.csv", index=False)

    with open(f"{OUT_DIR}/config.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "protocol": {
                    "description": "Repeated train/validation/test block evaluation with 12-week validation and 12-week test windows.",
                    "test_starts": TEST_STARTS,
                    "window_length_weeks": WINDOW,
                },
                "baseline": {
                    "weight_rule": "Select hybrid weight on validation from {0.70, 0.80, 0.90}",
                    "gb_params": {
                        "n_estimators": 500,
                        "learning_rate": 0.05,
                        "max_depth": 3,
                    },
                },
                "transformer": {
                    "pairs": [
                        "PatchTST+Transformer",
                        "PatchTST+Transformer+ElasticNetRoR",
                        "PatchTST+iTransformer",
                    ],
                    "reestimate": {
                        "seeds": TRANSFORMER_SEEDS,
                        "epochs": TRANSFORMER_EPOCHS,
                        "patience": TRANSFORMER_PATIENCE,
                    },
                    "ror": {
                        "alpha": ELASTIC_ALPHA,
                        "l1_ratio": ELASTIC_L1,
                        "lambda": ELASTIC_LAMBDA,
                        "gate": "Apply lambda only when validation RMSE improves over stage 2",
                    },
                },
                "stl": {
                    "baseline": "ExponentialSmoothing one-step expanding refit",
                    "nlinear": {
                        "seq_len": STL_SEQ_LEN,
                        "seeds": STL_SEEDS,
                        "topk": STL_TOPK,
                        "epochs": STL_EPOCHS,
                        "patience": STL_PATIENCE,
                    },
                    "ror": {
                        "model": "LightGBM",
                        "gate": "Apply stage 3 only when validation RMSE improves over stage 2",
                    },
                },
            },
            fp,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 72)
    print("Oil Common-Protocol Rolling Backtest")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print()
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
