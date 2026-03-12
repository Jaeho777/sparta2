"""
Direct oil adaptation of the nickel-style hybrid baseline.

Pipeline
1. One-week lag all exogenous variables to prevent contemporaneous leakage.
2. Fit a GradientBoostingRegressor on the training split.
3. Build a nickel-style hybrid forecast:
       y_hat = w * TwoPointLinear + (1 - w) * GB
4. Pick the best weight on validation and report untouched test metrics.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = "data_weekly_260120.csv"
TARGET = "Com_BrentCrudeOil"
VAL_START = "2025-08-04"
TEST_START = "2025-10-27"
OUT_DIR = "output_oil_nickel_style"
RANDOM_STATE = 42
HYBRID_WEIGHTS = [0.70, 0.80, 0.90]


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    two_point_val: np.ndarray
    two_point_test: np.ndarray


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").set_index("dt")
    df.index.freq = "W-MON"
    return df


def calc_two_point_linear(y: pd.Series, indices: pd.Index) -> np.ndarray:
    preds = []
    for idx in indices:
        loc = y.index.get_loc(idx)
        prev_1 = float(y.iloc[loc - 1])
        prev_2 = float(y.iloc[loc - 2])
        preds.append(prev_1 + (prev_1 - prev_2))
    return np.asarray(preds, dtype=float)


def prepare_splits(df: pd.DataFrame) -> SplitData:
    feature_cols = [c for c in df.columns if c != TARGET]
    X_all = df[feature_cols].shift(1)
    y_all = df[TARGET].copy()

    mask_train = df.index < VAL_START
    mask_val = (df.index >= VAL_START) & (df.index < TEST_START)
    mask_test = df.index >= TEST_START

    X_train = X_all.loc[mask_train]
    X_val = X_all.loc[mask_val]
    X_test = X_all.loc[mask_test]
    y_train = y_all.loc[mask_train]
    y_val = y_all.loc[mask_val]
    y_test = y_all.loc[mask_test]

    two_point_val = calc_two_point_linear(y_all, y_val.index)
    two_point_test = calc_two_point_linear(y_all, y_test.index)

    return SplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        two_point_val=two_point_val,
        two_point_test=two_point_test,
    )


def fit_gb(split: SplitData) -> tuple[GradientBoostingRegressor, np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(split.X_train)
    X_val = imputer.transform(split.X_val)
    X_test = imputer.transform(split.X_test)

    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, split.y_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    return model, val_pred, test_pred


def build_result(model_name: str, y_true: pd.Series, pred: np.ndarray) -> dict[str, float | str]:
    yt = y_true.to_numpy(dtype=float)
    return {
        "Model": model_name,
        "RMSE": rmse(yt, pred),
        "MAE": float(mean_absolute_error(yt, pred)),
        "MAPE(%)": mape(yt, pred),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_frame()
    split = prepare_splits(df)
    model, gb_val, gb_test = fit_gb(split)

    results = []

    results.append(build_result("TwoPointLinear", split.y_test, split.two_point_test))
    results.append(build_result("GradientBoosting", split.y_test, gb_test))

    best_weight = None
    best_val_rmse = float("inf")
    best_val_pred = None
    best_test_pred = None

    for w in HYBRID_WEIGHTS:
        val_pred = w * split.two_point_val + (1.0 - w) * gb_val
        test_pred = w * split.two_point_test + (1.0 - w) * gb_test
        val_rmse = rmse(split.y_val.to_numpy(dtype=float), val_pred)
        label = f"Hybrid_TwoPointLinear{w:.1f}_GB{1.0-w:.1f}"
        row = build_result(label, split.y_test, test_pred)
        row["Validation_RMSE"] = val_rmse
        results.append(row)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_weight = w
            best_val_pred = val_pred
            best_test_pred = test_pred

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUT_DIR}/results.csv", index=False)

    feature_importance = (
        pd.DataFrame(
            {
                "feature": split.X_train.columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)

    if best_weight is None or best_val_pred is None or best_test_pred is None:
        raise RuntimeError("Failed to choose a hybrid weight.")

    pd.DataFrame(
        {
            "dt": split.y_test.index,
            "actual": split.y_test.to_numpy(dtype=float),
            "two_point_linear": split.two_point_test,
            "gradient_boosting": gb_test,
            f"hybrid_{best_weight:.2f}": best_test_pred,
        }
    ).to_csv(f"{OUT_DIR}/test_predictions.csv", index=False)

    config = {
        "target": TARGET,
        "split": {
            "train_end": str(split.y_train.index[-1].date()),
            "val_start": VAL_START,
            "val_end": str(split.y_val.index[-1].date()),
            "test_start": TEST_START,
            "test_end": str(split.y_test.index[-1].date()),
        },
        "n_features": int(split.X_train.shape[1]),
        "best_weight_by_validation": best_weight,
        "best_validation_rmse": best_val_rmse,
    }
    with open(f"{OUT_DIR}/config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Oil Nickel-Style Hybrid Baseline")
    print("=" * 72)
    print(f"Target          : {TARGET}")
    print(f"Train / Val / Test: {len(split.y_train)} / {len(split.y_val)} / {len(split.y_test)}")
    print(f"Best hybrid w   : {best_weight:.2f}")
    print(f"Best val RMSE   : {best_val_rmse:.4f}")
    print()
    print(results_df.sort_values("RMSE").to_string(index=False))
    print()
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
