from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import oil_transformer_advanced_clean as tfm


ROOT = Path(__file__).resolve().parent
BASE_OUT = ROOT / "output_oil_nickel_style"
TFM_OUT = ROOT / "output_oil_transformer_clean"
STL_OUT = ROOT / "output_oil_academic"
COMMON_OUT = ROOT / "output_oil_common_protocol"
TARGET = "Com_BrentCrudeOil"


def ensure_dirs() -> None:
    for path in [BASE_OUT, TFM_OUT, STL_OUT, COMMON_OUT]:
        path.mkdir(parents=True, exist_ok=True)


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data_weekly_260120.csv")
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").set_index("dt")
    df.index.freq = "W-MON"
    return df


def build_baseline_figures() -> None:
    preds = pd.read_csv(BASE_OUT / "test_predictions.csv")
    feat = pd.read_csv(BASE_OUT / "feature_importance.csv").head(15)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(preds["dt"], preds["actual"], "k-o", lw=2, ms=4, label="Actual")
    ax.plot(preds["dt"], preds["two_point_linear"], ls="--", lw=1.5, label="Two-Point Linear")
    ax.plot(preds["dt"], preds["gradient_boosting"], ls=":", lw=1.8, label="GradientBoosting")
    hybrid_col = [c for c in preds.columns if c.startswith("hybrid_")][0]
    ax.plot(preds["dt"], preds[hybrid_col], "r-s", lw=2, ms=4, label=f"Best Hybrid ({hybrid_col})")
    ax.set_title("Baseline Notebook: Test Actual vs Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/barrel")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(BASE_OUT / "01_test_predictions.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feat["feature"][::-1], feat["importance"][::-1], color="#2d6a4f")
    ax.set_title("Baseline Notebook: GradientBoosting Feature Importance (Top 15)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(BASE_OUT / "02_feature_importance.png", bbox_inches="tight")
    plt.close(fig)


def _build_transformer_bundle() -> tfm.SequenceBundle:
    tfm.set_seed(tfm.SEED)
    df = load_frame()
    return tfm.prepare_sequences(df)


def _rerun_transformer_pair(
    bundle: tfm.SequenceBundle,
    base_name: str,
    resid_name: str,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, np.ndarray | str | float]]:
    row, artifact = tfm.run_stage2_experiment(
        bundle,
        base_name,
        resid_name,
        device,
        n_seeds=tfm.CONFIRM_STAGE_SEEDS,
        epochs=tfm.CONFIRM_EPOCHS,
        patience=tfm.CONFIRM_PATIENCE,
        keep_artifacts=True,
    )
    if artifact is None:
        raise RuntimeError(f"Missing artifacts for {base_name}+{resid_name}")
    return row, artifact


def _build_ror_inputs(
    bundle: tfm.SequenceBundle,
    artifact: dict[str, np.ndarray | str | float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_val = bundle.y_val.to_numpy(dtype=float)[-len(bundle.yva_seq) :]
    y_test = bundle.y_test.to_numpy(dtype=float)[-len(bundle.yte_seq) :]
    ror_ytr = artifact["resid_train_target"] - artifact["resid_train"]
    ror_Xtr = bundle.Xtr_sc[tfm.SEQ_LEN:][: len(ror_ytr)]
    usable = min(len(ror_ytr), len(ror_Xtr))
    ror_ytr = ror_ytr[:usable]
    ror_Xtr = ror_Xtr[:usable]
    ror_Xva = bundle.Xva_sc[-len(bundle.yva_seq) :]
    ror_Xte = bundle.Xte_sc[-len(bundle.yte_seq) :]
    return (
        ror_Xtr,
        ror_ytr,
        ror_Xva,
        ror_Xte,
        artifact["s2_val"],
        artifact["s2_test"],
        y_val,
        y_test,
    )


def build_transformer_figures() -> None:
    shap_df = pd.read_csv(TFM_OUT / "feature_shap.csv").head(15)
    cv_df = pd.read_csv(TFM_OUT / "feature_selection_cv.csv")
    stage2_df = pd.read_csv(TFM_OUT / "stage2_experiments.csv")
    ror_df = pd.read_csv(TFM_OUT / "top3_ror_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(shap_df["feature"][::-1], shap_df["mean_abs_shap"][::-1], color="#264653")
    axes[0].set_title("Train-only SHAP Importance (Top 15)")
    axes[0].set_xlabel("Mean |SHAP|")
    axes[1].plot(cv_df["n_features"], cv_df["cv_rmse"], "o-", color="#e76f51", lw=2)
    axes[1].set_title("Feature Count Selection by TimeSeriesCV")
    axes[1].set_xlabel("Number of Features")
    axes[1].set_ylabel("CV RMSE")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(TFM_OUT / "01_feature_selection_summary.png", bbox_inches="tight")
    plt.close(fig)

    top8 = stage2_df.copy()
    x = np.arange(len(top8))
    width = 0.36
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, top8["S2_Val_RMSE"], width, label="Validation RMSE", color="#457b9d")
    ax.bar(x + width / 2, top8["S2_Test_RMSE"], width, label="Test RMSE", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(top8["Experiment"], rotation=35, ha="right")
    ax.set_title("Transformer Notebook: Confirmatory Top-8 Stage-2 Comparison")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(TFM_OUT / "02_top8_stage2_rmse.png", bbox_inches="tight")
    plt.close(fig)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = _build_transformer_bundle()
    selected_row, selected_art = _rerun_transformer_pair(bundle, "PatchTST", "Transformer", device)
    holdout_row, holdout_art = _rerun_transformer_pair(bundle, "PatchTST", "iTransformer", device)

    (
        selected_ror_Xtr,
        selected_ror_ytr,
        selected_ror_Xva,
        selected_ror_Xte,
        selected_s2_val,
        selected_s2_test,
        selected_y_val,
        selected_y_test,
    ) = _build_ror_inputs(bundle, selected_art)

    final_val, final_test, _, final_desc = tfm.ror_elasticnet(
        selected_ror_Xtr,
        selected_ror_ytr,
        selected_ror_Xva,
        selected_ror_Xte,
        selected_s2_val,
        selected_s2_test,
        selected_y_val,
        selected_y_test,
        bundle.ystd,
    )

    dt_test = bundle.y_test.index[-len(selected_y_test) :]
    prediction_df = pd.DataFrame(
        {
            "dt": dt_test,
            "actual": selected_y_test,
            "selected_base_patchtst": tfm.safe_to_price(selected_art["base_test"], bundle.ymean, bundle.ystd),
            "selected_stage2_patchtst_transformer": selected_s2_test,
            "selected_final_patchtst_transformer_elasticnet": final_test,
            "holdout_stage2_patchtst_itransformer": holdout_art["s2_test"],
        }
    )
    prediction_df.to_csv(TFM_OUT / "selected_model_test_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(prediction_df["dt"], prediction_df["actual"], "k-o", lw=2, ms=4, label="Actual")
    ax.plot(
        prediction_df["dt"],
        prediction_df["selected_stage2_patchtst_transformer"],
        ls="--",
        lw=1.8,
        label="PatchTST+Transformer (Stage 2)",
    )
    ax.plot(
        prediction_df["dt"],
        prediction_df["selected_final_patchtst_transformer_elasticnet"],
        "r-s",
        lw=2,
        ms=4,
        label=f"PatchTST+Transformer+ElasticNet ({final_desc})",
    )
    ax.plot(
        prediction_df["dt"],
        prediction_df["holdout_stage2_patchtst_itransformer"],
        color="#2a9d8f",
        ls="-.",
        lw=2,
        label="PatchTST+iTransformer (lowest holdout Stage 2)",
    )
    ax.set_title("Transformer Notebook: Test Actual vs Key Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/barrel")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(TFM_OUT / "03_test_predictions_key_models.png", bbox_inches="tight")
    plt.close(fig)

    top3 = ror_df[["Experiment", "S2_Test_RMSE", "Final_Test_RMSE"]].copy()
    x = np.arange(len(top3))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, top3["S2_Test_RMSE"], width, label="Stage 2 Test RMSE", color="#577590")
    ax.bar(x + width / 2, top3["Final_Test_RMSE"], width, label="Final Test RMSE", color="#f3722c")
    ax.set_xticks(x)
    ax.set_xticklabels(top3["Experiment"], rotation=20, ha="right")
    ax.set_title("Transformer Notebook: Top-3 Stage 2 vs Final Test RMSE")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(TFM_OUT / "04_top3_stage2_vs_final.png", bbox_inches="tight")
    plt.close(fig)


def build_common_protocol_figure() -> None:
    df = pd.read_csv(COMMON_OUT / "rolling_origin_summary.csv")
    fig, ax = plt.subplots(figsize=(12, 6))
    ordered = df.sort_values("mean_test_rmse")
    ax.barh(ordered["model"], ordered["mean_test_rmse"], color="#4d908e")
    ax.set_title("Common Repeated Evaluation: Mean Test RMSE")
    ax.set_xlabel("Mean Test RMSE")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(COMMON_OUT / "01_mean_test_rmse.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    ensure_dirs()
    build_baseline_figures()
    build_transformer_figures()
    build_common_protocol_figure()
    print("Oil visual assets created.")


if __name__ == "__main__":
    main()
