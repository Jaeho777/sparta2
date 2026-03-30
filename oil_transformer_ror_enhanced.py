"""
Brent Oil Forecasting: Enhanced RoR (Residual-of-Residual) Experiments
======================================================================
Pipeline: Baseline(DL) → Residual(DL) → Enhanced RoR

RoR Strategies:
  A. Weighted LightGBM (λ grid search 0.0~0.5)
  B. Ridge Regression (simple linear, anti-overfit)
  C. ElasticNet (sparse linear)
  D. LightGBM + Expanding Window OOF
  E. Feature-Augmented LightGBM (residual stats + base predictions)
  F. Ensemble RoR (Ridge + LightGBM blend)
  G. Random Walk Blend (shrinkage toward RW prior)
  H. All-Feature LightGBM (55 features for RoR)
"""
import os, json, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import shap
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
OUT = "output_oil_ror_enhanced"
os.makedirs(OUT, exist_ok=True)
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. DATA ─────────────────────────────────────────────────────────────
print("=" * 70)
print("  Phase 1: Data & Feature Engineering")
print("=" * 70)

df = pd.read_csv("data_weekly_260120.csv")
df["dt"] = pd.to_datetime(df["dt"])
df = df.sort_values("dt").set_index("dt")
df.index.freq = "W-MON"
TARGET = "Com_BrentCrudeOil"

VAL_START, TEST_START = "2025-08-04", "2025-10-27"
m_tr = df.index < VAL_START
m_va = (df.index >= VAL_START) & (df.index < TEST_START)
m_te = df.index >= TEST_START
y = df[TARGET].copy()
y_tr, y_va, y_te = y[m_tr], y[m_va], y[m_te]
print(f"  Train={m_tr.sum()} Val={m_va.sum()} Test={m_te.sum()}")

# Domain features
BASE = ["Com_Gasoline","Com_NaturalGas","Com_Uranium","Com_Coal",
        "Com_LME_Cu_Cash","Com_Steel","Com_Iron_Ore",
        "Idx_DxyUSD","EX_USD_CNY",
        "Bonds_US_10Y","Bonds_US_2Y","Bonds_US_3M",
        "Idx_SnPVIX","Com_Gold","Idx_SnP500","Idx_CSI300",
        "EX_USD_KRW","Bonds_KOR_10Y","EX_USD_JPY",
        "Com_Corn","Com_Soybeans","Com_PalmOil"]
BASE = [c for c in BASE if c in df.columns]

dfd = pd.DataFrame(index=df.index)
for c in BASE:
    dfd[f"{c}_ret"] = df[c].pct_change()
dfd["Spread_10Y_2Y"] = df["Bonds_US_10Y"] - df["Bonds_US_2Y"]
dfd["Spread_Crack"] = df["Com_Gasoline"] - df[TARGET]
dfd["Ratio_Gold_Oil"] = df["Com_Gold"] / df[TARGET]
for c in ["Com_Gasoline","Com_NaturalGas","Idx_SnPVIX","Idx_DxyUSD"]:
    dfd[f"{c}_ma4r"] = df[c] / df[c].rolling(4).mean() - 1
    dfd[f"{c}_ma12r"] = df[c] / df[c].rolling(12).mean() - 1

ALL_FEAT = BASE + list(dfd.columns)
X_lag = pd.concat([df[BASE], dfd], axis=1).shift(1)  # 1-week lag
print(f"  Features: {len(BASE)} base + {len(dfd.columns)} derived = {len(ALL_FEAT)}")

# ── 2. SHAP FEATURE SELECTION ───────────────────────────────────────────
print("\n" + "=" * 70)
print("  Phase 2: SHAP + TimeSeriesCV Feature Selection")
print("=" * 70)

Xtr_raw = np.nan_to_num(X_lag[m_tr].values, nan=0.0)
ytr_arr = y_tr.values

lgb_shap = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
    num_leaves=31, verbose=-1, random_state=SEED)
lgb_shap.fit(Xtr_raw, ytr_arr)

print("  Computing SHAP values...")
explainer = shap.TreeExplainer(lgb_shap)
sv = explainer.shap_values(Xtr_raw)
mean_shap = np.abs(sv).mean(axis=0)
shap_df = pd.DataFrame({"feature": ALL_FEAT, "shap": mean_shap}
    ).sort_values("shap", ascending=False)

print("  SHAP Top 15:")
for _, r in shap_df.head(15).iterrows():
    print(f"    {r['feature']:30s} {r['shap']:.4f}")

# CV for optimal count
ranked = shap_df["feature"].tolist()
tscv = TimeSeriesSplit(n_splits=5)
cv_res = []
for nf in [10, 15, 20, 25, 30, 40, 55]:
    idx = [ALL_FEAT.index(f) for f in ranked[:nf]]
    Xs = Xtr_raw[:, idx]
    rmses = []
    for ti, vi in tscv.split(Xs):
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
            num_leaves=20, verbose=-1, random_state=SEED)
        m.fit(Xs[ti], ytr_arr[ti])
        rmses.append(np.sqrt(mean_squared_error(ytr_arr[vi], m.predict(Xs[vi]))))
    cv_res.append({"n": nf, "cv": np.mean(rmses)})
    print(f"    n={nf:3d} CV_RMSE={np.mean(rmses):.4f}")

cv_df = pd.DataFrame(cv_res)
best_n = int(cv_df.loc[cv_df["cv"].idxmin(), "n"])
SEL_FEAT = ranked[:best_n]
print(f"  * Optimal: {best_n} features (CV={cv_df['cv'].min():.4f})")

shap_df.to_csv(f"{OUT}/shap_importance.csv", index=False)

# ── 3. PREPARE SEQUENCES ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Phase 3: Sequence Preparation")
print("=" * 70)

SEQ_LEN = 24
sel_idx = [ALL_FEAT.index(f) for f in SEL_FEAT]
Xtr_s = np.nan_to_num(X_lag[m_tr].values[:, sel_idx], nan=0.0)
Xva_s = np.nan_to_num(X_lag[m_va].values[:, sel_idx], nan=0.0)
Xte_s = np.nan_to_num(X_lag[m_te].values[:, sel_idx], nan=0.0)

scaler = RobustScaler()
Xtr_sc = scaler.fit_transform(Xtr_s)
Xva_sc = scaler.transform(Xva_s)
Xte_sc = scaler.transform(Xte_s)

# Also prepare ALL features (for RoR feature augmentation)
Xtr_all = np.nan_to_num(X_lag[m_tr].values, nan=0.0)
Xva_all = np.nan_to_num(X_lag[m_va].values, nan=0.0)
Xte_all = np.nan_to_num(X_lag[m_te].values, nan=0.0)
scaler_all = RobustScaler()
Xtr_all_sc = scaler_all.fit_transform(Xtr_all)
Xva_all_sc = scaler_all.transform(Xva_all)
Xte_all_sc = scaler_all.transform(Xte_all)

ymean, ystd = y_tr.mean(), y_tr.std()
ytr_n = (y_tr.values - ymean) / ystd
yva_n = (y_va.values - ymean) / ystd
yte_n = (y_te.values - ymean) / ystd

def mkseq(X, y, sl):
    Xs, ys = [], []
    for i in range(sl, len(X)):
        Xs.append(X[i-sl:i]); ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

Xtr_seq, ytr_seq = mkseq(Xtr_sc, ytr_n, SEQ_LEN)
buf_X = Xtr_sc[-SEQ_LEN:]
buf_y = ytr_n[-SEQ_LEN:]
Xva_seq, yva_seq = mkseq(np.vstack([buf_X, Xva_sc]),
                          np.concatenate([buf_y, yva_n]), SEQ_LEN)
all_before_test_X = np.vstack([Xtr_sc, Xva_sc])
all_before_test_y = np.concatenate([ytr_n, yva_n])
te_buf_X = all_before_test_X[-SEQ_LEN:]
te_buf_y = all_before_test_y[-SEQ_LEN:]
Xte_seq, yte_seq = mkseq(np.vstack([te_buf_X, Xte_sc]),
                          np.concatenate([te_buf_y, yte_n]), SEQ_LEN)

n_feat = Xtr_seq.shape[2]
print(f"  Features: {n_feat}, Train seq: {Xtr_seq.shape}, Val: {Xva_seq.shape}, Test: {Xte_seq.shape}")

tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr_seq),
    torch.from_numpy(ytr_seq)), batch_size=32, shuffle=True)
va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva_seq),
    torch.from_numpy(yva_seq)), batch_size=32, shuffle=False)

# ── 4. MODELS ────────────────────────────────────────────────────────────
class NLinearM(nn.Module):
    def __init__(self, sl, nf, dh=64, do=0.3):
        super().__init__()
        self.l1 = nn.Linear(sl, dh)
        self.l2 = nn.Linear(nf, dh)
        self.out = nn.Linear(dh*2, 1)
        self.do = nn.Dropout(do); self.act = nn.ReLU()
    def forward(self, x):
        xm = x[:,:,0]; xm = xm - xm[:,-1:]
        h1 = self.act(self.l1(xm))
        h2 = self.act(self.l2(x[:,-1,:]))
        return self.out(self.do(torch.cat([h1,h2],-1))).squeeze(-1)

class PatchTSTM(nn.Module):
    def __init__(self, sl, nf, dm=64, nh=4, nl=2, pl=4, st=2, do=0.2):
        super().__init__()
        self.pl, self.st = pl, st
        np_ = (sl - pl) // st + 1
        self.proj = nn.Linear(pl * nf, dm)
        self.pos = nn.Parameter(torch.randn(1, np_+1, dm)*0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, dm)*0.02)
        el = nn.TransformerEncoderLayer(dm, nh, dm*4, do, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(el, nl)
        self.head = nn.Linear(dm, 1); self.do = nn.Dropout(do)
    def forward(self, x):
        B = x.size(0)
        ps = [x[:,i:i+self.pl,:].reshape(B,-1) for i in range(0,x.size(1)-self.pl+1,self.st)]
        z = self.proj(torch.stack(ps, 1))
        z = torch.cat([self.cls.expand(B,-1,-1), z], 1) + self.pos
        z = self.enc(z)
        return self.head(self.do(z[:,0])).squeeze(-1)

class iTransM(nn.Module):
    def __init__(self, sl, nf, dm=64, nh=4, nl=2, do=0.2):
        super().__init__()
        self.proj = nn.Linear(sl, dm)
        self.pos = nn.Parameter(torch.randn(1, nf, dm)*0.02)
        el = nn.TransformerEncoderLayer(dm, nh, dm*4, do, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(el, nl)
        self.head = nn.Sequential(nn.Linear(nf*dm, dm), nn.GELU(), nn.Dropout(do), nn.Linear(dm,1))
    def forward(self, x):
        z = self.proj(x.permute(0,2,1)) + self.pos
        z = self.enc(z).reshape(x.size(0), -1)
        return self.head(z).squeeze(-1)

class TransM(nn.Module):
    def __init__(self, sl, nf, dm=64, nh=4, nl=2, do=0.2):
        super().__init__()
        self.proj = nn.Linear(nf, dm)
        self.pos = nn.Parameter(torch.randn(1, sl, dm)*0.02)
        el = nn.TransformerEncoderLayer(dm, nh, dm*4, do, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(el, nl)
        self.head = nn.Sequential(nn.Linear(dm, dm), nn.GELU(), nn.Dropout(do), nn.Linear(dm,1))
    def forward(self, x):
        z = self.proj(x) + self.pos
        return self.head(self.enc(z)[:,-1]).squeeze(-1)

class LSTMM(nn.Module):
    def __init__(self, nf, hs=64, nl=2, do=0.2):
        super().__init__()
        self.lstm = nn.LSTM(nf, hs, nl, dropout=do if nl>1 else 0, batch_first=True)
        self.head = nn.Linear(hs, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:,-1]).squeeze(-1)

MODELS = {
    "NLinear": lambda: NLinearM(SEQ_LEN, n_feat),
    "PatchTST": lambda: PatchTSTM(SEQ_LEN, n_feat),
    "iTransformer": lambda: iTransM(SEQ_LEN, n_feat),
    "Transformer": lambda: TransM(SEQ_LEN, n_feat),
    "LSTM": lambda: LSTMM(n_feat),
}

# ── 5. TRAINING UTIL ─────────────────────────────────────────────────────
def train_one(model, tr_ld, va_ld, epochs=150, patience=25, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    crit = nn.MSELoss(); best_vl = 1e9; best_st = None; wait = 0
    for ep in range(epochs):
        model.train(); tl = 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item()*len(xb)
        model.eval(); vl = 0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(device), yb.to(device)
                vl += crit(model(xb), yb).item()*len(xb)
        vl /= len(va_ld.dataset); sch.step(vl)
        if vl < best_vl:
            best_vl = vl; best_st = {k:v.cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait += 1
            if wait >= patience: break
    if best_st: model.load_state_dict(best_st)
    model.eval()
    return model, np.sqrt(best_vl)*ystd, ep+1

def pred(model, Xseq):
    model.eval()
    if len(Xseq) == 0:
        return np.array([], dtype=np.float32)
    with torch.no_grad():
        return model(torch.from_numpy(Xseq).to(device)).cpu().numpy()

def to_price(p): return p * ystd + ymean

# ── 6. ENHANCED RoR STRATEGIES ──────────────────────────────────────────

def build_ror_features(Xflat, resid_pred_hist=None, base_pred=None, stage2_pred=None):
    """Build augmented features for RoR stage.

    Adds:
    - Original scaled features
    - Stage 1 & 2 predictions (model state awareness)
    - Rolling statistics of recent residual predictions
    """
    parts = [Xflat]
    if base_pred is not None:
        parts.append(base_pred.reshape(-1, 1))
    if stage2_pred is not None:
        parts.append(stage2_pred.reshape(-1, 1))
    return np.hstack(parts)


LAMBDA_GRID = np.arange(0.0, 0.52, 0.02)  # Fine-grained lambda grid

def ror_strategy_weighted_lgbm(ror_Xtr, ror_tr, ror_Xva, ror_Xte, s2_va, s2_te, y_va_arr, y_te_arr):
    """Strategy A: LightGBM with optimal lambda weight search."""
    lgb_r = lgb.LGBMRegressor(
        learning_rate=0.01, num_leaves=7, min_child_samples=50,
        subsample=0.5, colsample_bytree=0.4,
        reg_alpha=5.0, reg_lambda=5.0, n_estimators=200,
        verbose=-1, random_state=SEED
    )
    lgb_r.fit(ror_Xtr, ror_tr)
    ror_va_pred = lgb_r.predict(ror_Xva)
    ror_te_pred = lgb_r.predict(ror_Xte)

    best_lam, best_va_rmse = 0.0, 1e9
    for lam in LAMBDA_GRID:
        s3_va = s2_va + lam * ror_va_pred * ystd
        va_rmse = np.sqrt(mean_squared_error(y_va_arr, s3_va))
        if va_rmse < best_va_rmse:
            best_lam, best_va_rmse = lam, va_rmse

    s3_va = s2_va + best_lam * ror_va_pred * ystd
    s3_te = s2_te + best_lam * ror_te_pred * ystd
    return s3_va, s3_te, best_va_rmse, f"WeightedLGBM(l={best_lam:.2f})"


def ror_strategy_ridge(ror_Xtr, ror_tr, ror_Xva, ror_Xte, s2_va, s2_te, y_va_arr, y_te_arr):
    """Strategy B: Ridge regression (simple, anti-overfit)."""
    best_alpha, best_va_rmse, best_va, best_te = 1.0, 1e9, None, None
    best_lam = 0.0

    for alpha in [0.1, 1.0, 10.0, 50.0, 100.0, 500.0]:
        ridge = Ridge(alpha=alpha, random_state=SEED)
        ridge.fit(ror_Xtr, ror_tr)
        ror_va_pred = ridge.predict(ror_Xva)
        ror_te_pred = ridge.predict(ror_Xte)

        for lam in LAMBDA_GRID:
            s3_va = s2_va + lam * ror_va_pred * ystd
            va_rmse = np.sqrt(mean_squared_error(y_va_arr, s3_va))
            if va_rmse < best_va_rmse:
                best_alpha = alpha
                best_lam = lam
                best_va_rmse = va_rmse
                best_va = s3_va.copy()
                best_te = s2_te + lam * ror_te_pred * ystd

    return best_va, best_te, best_va_rmse, f"Ridge(a={best_alpha:.0f},l={best_lam:.2f})"


def ror_strategy_elasticnet(ror_Xtr, ror_tr, ror_Xva, ror_Xte, s2_va, s2_te, y_va_arr, y_te_arr):
    """Strategy C: ElasticNet (sparse linear)."""
    best_params, best_va_rmse, best_va, best_te = {}, 1e9, None, None

    for alpha in [0.01, 0.1, 1.0, 10.0]:
        for l1_ratio in [0.3, 0.5, 0.7, 0.9]:
            en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=SEED, max_iter=5000)
            en.fit(ror_Xtr, ror_tr)
            ror_va_pred = en.predict(ror_Xva)
            ror_te_pred = en.predict(ror_Xte)

            for lam in LAMBDA_GRID:
                s3_va = s2_va + lam * ror_va_pred * ystd
                va_rmse = np.sqrt(mean_squared_error(y_va_arr, s3_va))
                if va_rmse < best_va_rmse:
                    best_params = {"alpha": alpha, "l1": l1_ratio, "lam": lam}
                    best_va_rmse = va_rmse
                    best_va = s3_va.copy()
                    best_te = s2_te + lam * ror_te_pred * ystd

    return best_va, best_te, best_va_rmse, f"ElasticNet(a={best_params.get('alpha','?')},l={best_params.get('lam','?')})"


def ror_strategy_oof_lgbm(ror_Xtr, ror_tr, ror_Xva, ror_Xte, s2_va, s2_te, y_va_arr, y_te_arr):
    """Strategy D: LightGBM with expanding window OOF to prevent overfitting."""
    n = len(ror_tr)
    min_train = max(100, n // 3)
    n_folds = 5
    fold_size = (n - min_train) // n_folds

    # OOF predictions for training data
    oof_pred = np.zeros(n)
    oof_count = np.zeros(n)
    models = []

    for fold in range(n_folds):
        tr_end = min_train + fold * fold_size
        va_start = tr_end
        va_end = min(va_start + fold_size, n)
        if va_start >= n:
            break

        lgb_fold = lgb.LGBMRegressor(
            learning_rate=0.01, num_leaves=7, min_child_samples=50,
            subsample=0.5, colsample_bytree=0.4,
            reg_alpha=5.0, reg_lambda=5.0, n_estimators=200,
            verbose=-1, random_state=SEED + fold
        )
        lgb_fold.fit(ror_Xtr[:tr_end], ror_tr[:tr_end])
        oof_pred[va_start:va_end] = lgb_fold.predict(ror_Xtr[va_start:va_end])
        oof_count[va_start:va_end] = 1
        models.append(lgb_fold)

    # Final model on all training data
    lgb_final = lgb.LGBMRegressor(
        learning_rate=0.01, num_leaves=7, min_child_samples=50,
        subsample=0.5, colsample_bytree=0.4,
        reg_alpha=5.0, reg_lambda=5.0, n_estimators=200,
        verbose=-1, random_state=SEED
    )
    lgb_final.fit(ror_Xtr, ror_tr)

    # Ensemble prediction (average of fold models + final)
    va_preds = [m.predict(ror_Xva) for m in models] + [lgb_final.predict(ror_Xva)]
    te_preds = [m.predict(ror_Xte) for m in models] + [lgb_final.predict(ror_Xte)]
    ror_va_pred = np.mean(va_preds, axis=0)
    ror_te_pred = np.mean(te_preds, axis=0)

    best_lam, best_va_rmse = 0.0, 1e9
    for lam in LAMBDA_GRID:
        s3_va = s2_va + lam * ror_va_pred * ystd
        va_rmse = np.sqrt(mean_squared_error(y_va_arr, s3_va))
        if va_rmse < best_va_rmse:
            best_lam, best_va_rmse = lam, va_rmse

    s3_va = s2_va + best_lam * ror_va_pred * ystd
    s3_te = s2_te + best_lam * ror_te_pred * ystd
    return s3_va, s3_te, best_va_rmse, f"OOF_LGBM(l={best_lam:.2f},folds={len(models)})"


def ror_strategy_augmented_lgbm(ror_Xtr, ror_tr, ror_Xva, ror_Xte, s2_va, s2_te, y_va_arr, y_te_arr,
                                 base_tr=None, base_va=None, base_te=None,
                                 resid_tr=None, resid_va=None, resid_te=None):
    """Strategy E: Feature-augmented LightGBM (add base & residual predictions as features)."""
    # Augment features with model predictions
    Xtr_aug = build_ror_features(ror_Xtr, base_pred=base_tr, stage2_pred=resid_tr)
    Xva_aug = build_ror_features(ror_Xva, base_pred=base_va, stage2_pred=resid_va)
    Xte_aug = build_ror_features(ror_Xte, base_pred=base_te, stage2_pred=resid_te)

    lgb_r = lgb.LGBMRegressor(
        learning_rate=0.01, num_leaves=7, min_child_samples=50,
        subsample=0.5, colsample_bytree=0.4,
        reg_alpha=5.0, reg_lambda=5.0, n_estimators=200,
        verbose=-1, random_state=SEED
    )
    lgb_r.fit(Xtr_aug, ror_tr)
    ror_va_pred = lgb_r.predict(Xva_aug)
    ror_te_pred = lgb_r.predict(Xte_aug)

    best_lam, best_va_rmse = 0.0, 1e9
    for lam in LAMBDA_GRID:
        s3_va = s2_va + lam * ror_va_pred * ystd
        va_rmse = np.sqrt(mean_squared_error(y_va_arr, s3_va))
        if va_rmse < best_va_rmse:
            best_lam, best_va_rmse = lam, va_rmse

    s3_va = s2_va + best_lam * ror_va_pred * ystd
    s3_te = s2_te + best_lam * ror_te_pred * ystd
    return s3_va, s3_te, best_va_rmse, f"AugLGBM(l={best_lam:.2f})", lgb_r


def ror_strategy_ensemble(ror_Xtr, ror_tr, ror_Xva, ror_Xte, s2_va, s2_te, y_va_arr, y_te_arr):
    """Strategy F: Ridge + LightGBM ensemble blend."""
    # Ridge
    ridge = Ridge(alpha=50.0, random_state=SEED)
    ridge.fit(ror_Xtr, ror_tr)
    r_va = ridge.predict(ror_Xva)
    r_te = ridge.predict(ror_Xte)

    # LightGBM (conservative)
    lgb_r = lgb.LGBMRegressor(
        learning_rate=0.01, num_leaves=5, min_child_samples=60,
        subsample=0.5, colsample_bytree=0.4,
        reg_alpha=8.0, reg_lambda=8.0, n_estimators=150,
        verbose=-1, random_state=SEED
    )
    lgb_r.fit(ror_Xtr, ror_tr)
    l_va = lgb_r.predict(ror_Xva)
    l_te = lgb_r.predict(ror_Xte)

    best_w, best_lam, best_va_rmse = 0.5, 0.0, 1e9
    best_va, best_te = None, None
    for w in np.arange(0.0, 1.05, 0.1):  # weight for Ridge
        ror_va_blend = w * r_va + (1 - w) * l_va
        ror_te_blend = w * r_te + (1 - w) * l_te
        for lam in LAMBDA_GRID:
            s3_va = s2_va + lam * ror_va_blend * ystd
            va_rmse = np.sqrt(mean_squared_error(y_va_arr, s3_va))
            if va_rmse < best_va_rmse:
                best_w, best_lam, best_va_rmse = w, lam, va_rmse
                best_va = s3_va.copy()
                best_te = (s2_te + lam * ror_te_blend * ystd).copy()

    return best_va, best_te, best_va_rmse, f"Ensemble(w_ridge={best_w:.1f},l={best_lam:.2f})"


def ror_strategy_rw_blend(s2_va, s2_te, y_va_arr, y_te_arr, rw_va_arr, rw_te_arr):
    """Strategy G: Blend S2 prediction with Random Walk (shrinkage toward RW prior).

    The RW is a strong baseline. Blending shrinks predictions toward the prior,
    reducing variance at the cost of introducing slight bias.
    Final = (1-w) * S2 + w * RW, where w is optimized on validation.
    """
    best_w, best_va_rmse = 0.0, 1e9
    best_va, best_te = None, None

    for w in np.arange(0.0, 0.52, 0.02):
        s3_va = (1 - w) * s2_va + w * rw_va_arr[-len(s2_va):]
        va_rmse = np.sqrt(mean_squared_error(y_va_arr, s3_va))
        if va_rmse < best_va_rmse:
            best_w = w
            best_va_rmse = va_rmse
            best_va = s3_va.copy()
            best_te = ((1 - w) * s2_te + w * rw_te_arr[-len(s2_te):]).copy()

    return best_va, best_te, best_va_rmse, f"RW_Blend(w={best_w:.2f})"


def ror_strategy_allfeature_lgbm(ror_Xtr_all, ror_tr, ror_Xva_all, ror_Xte_all,
                                   s2_va, s2_te, y_va_arr, y_te_arr):
    """Strategy H: LightGBM using ALL 55 features for RoR (more information)."""
    lgb_r = lgb.LGBMRegressor(
        learning_rate=0.008, num_leaves=5, min_child_samples=60,
        subsample=0.4, colsample_bytree=0.3,
        reg_alpha=10.0, reg_lambda=10.0, n_estimators=150,
        verbose=-1, random_state=SEED
    )
    lgb_r.fit(ror_Xtr_all, ror_tr)
    ror_va_pred = lgb_r.predict(ror_Xva_all)
    ror_te_pred = lgb_r.predict(ror_Xte_all)

    best_lam, best_va_rmse = 0.0, 1e9
    for lam in LAMBDA_GRID:
        s3_va = s2_va + lam * ror_va_pred * ystd
        va_rmse = np.sqrt(mean_squared_error(y_va_arr, s3_va))
        if va_rmse < best_va_rmse:
            best_lam, best_va_rmse = lam, va_rmse

    s3_va = s2_va + best_lam * ror_va_pred * ystd
    s3_te = s2_te + best_lam * ror_te_pred * ystd
    return s3_va, s3_te, best_va_rmse, f"AllFeatLGBM(l={best_lam:.2f})"


# ── 7. TOP EXPERIMENTS WITH ENHANCED RoR ─────────────────────────────────
print("\n" + "=" * 70)
print("  Phase 5: Enhanced RoR Experiments")
print("=" * 70)

# Focus on 5 most promising combinations
EXPS = [
    ("PatchTST",     "iTransformer"),   # Exp9 - was best
    ("LSTM",         "LSTM"),            # Exp5 - was 2nd
    ("PatchTST",     "PatchTST"),        # Exp2 - was 3rd
    ("Transformer",  "Transformer"),     # Exp4 - was 4th
    ("PatchTST",     "NLinear"),         # Exp6
]

# Random Walk
rw_va = np.array([y.iloc[y.index.get_loc(i)-1] for i in y_va.index])
rw_te = np.array([y.iloc[y.index.get_loc(i)-1] for i in y_te.index])
rw_rmse = np.sqrt(mean_squared_error(y_te, rw_te))
print(f"  Random Walk: Test RMSE={rw_rmse:.4f}")

N_SEEDS = 5
results = []
all_ror_details = []

for ei, (bn, rn) in enumerate(EXPS):
    label = f"Exp{ei+1}: {bn}+{rn}"
    t0 = time.time()
    print(f"\n  {'='*60}")
    print(f"  {label}")
    print(f"  {'='*60}")

    # Stage 1: Baseline (3-seed ensemble)
    bv, bt, btr = [], [], []
    for si in range(N_SEEDS):
        torch.manual_seed(SEED+si); np.random.seed(SEED+si)
        m, vr, ep = train_one(MODELS[bn](), tr_loader, va_loader)
        bv.append(pred(m, Xva_seq)); bt.append(pred(m, Xte_seq))
        btr.append(pred(m, Xtr_seq))
    bv_e, bt_e, btr_e = np.mean(bv,0), np.mean(bt,0), np.mean(btr,0)
    bv_p, bt_p = to_price(bv_e), to_price(bt_e)

    b_va_rmse = np.sqrt(mean_squared_error(y_va.values[-len(bv_p):], bv_p))
    b_te_rmse = np.sqrt(mean_squared_error(y_te.values[-len(bt_p):], bt_p))
    print(f"    Stage1 Base({bn}): Val={b_va_rmse:.4f} Test={b_te_rmse:.4f}")

    # Stage 2: Residual
    resid_tr_tgt = (ytr_seq - btr_e).astype(np.float32)
    resid_va_tgt = (yva_seq - bv_e[-len(yva_seq):]).astype(np.float32)
    r_tr_ld = DataLoader(TensorDataset(torch.from_numpy(Xtr_seq),
        torch.from_numpy(resid_tr_tgt)), batch_size=32, shuffle=True)
    r_va_ld = DataLoader(TensorDataset(torch.from_numpy(Xva_seq),
        torch.from_numpy(resid_va_tgt)), batch_size=32, shuffle=False)

    rv, rt, rtr = [], [], []
    for si in range(N_SEEDS):
        torch.manual_seed(SEED+100+si); np.random.seed(SEED+100+si)
        m, vr, ep = train_one(MODELS[rn](), r_tr_ld, r_va_ld)
        rv.append(pred(m, Xva_seq)); rt.append(pred(m, Xte_seq))
        rtr.append(pred(m, Xtr_seq))
    rv_e, rt_e, rtr_e = np.mean(rv,0), np.mean(rt,0), np.mean(rtr,0)

    s2_va = to_price(bv_e[-len(rv_e):] + rv_e)
    s2_te = to_price(bt_e[-len(rt_e):] + rt_e)
    s2_va_rmse = np.sqrt(mean_squared_error(y_va.values[-len(s2_va):], s2_va))
    s2_te_rmse = np.sqrt(mean_squared_error(y_te.values[-len(s2_te):], s2_te))
    print(f"    Stage2 +Resid({rn}): Val={s2_va_rmse:.4f} Test={s2_te_rmse:.4f}")

    # Stage 3: Enhanced RoR - Try ALL strategies
    ror_tr = resid_tr_tgt - rtr_e
    ror_Xtr = Xtr_sc[SEQ_LEN:][:len(ror_tr)]
    ror_Xva = Xva_sc[-len(yva_seq):]
    ror_Xte = Xte_sc[-len(yte_seq):]
    mn = min(len(ror_tr), len(ror_Xtr))
    ror_tr_c, ror_Xtr_c = ror_tr[:mn], ror_Xtr[:mn]

    # ALL features for Strategy H
    ror_Xtr_all = Xtr_all_sc[SEQ_LEN:][:mn]
    ror_Xva_all = Xva_all_sc[-len(yva_seq):]
    ror_Xte_all = Xte_all_sc[-len(yte_seq):]

    y_va_arr = y_va.values[-len(s2_va):]
    y_te_arr = y_te.values[-len(s2_te):]

    # Prepare augmented features for Strategy E
    base_tr_flat = btr_e[:mn]
    base_va_flat = bv_e[-len(yva_seq):]
    base_te_flat = bt_e[-len(yte_seq):]
    resid_tr_flat = rtr_e[:mn]
    resid_va_flat = rv_e[-len(yva_seq):]
    resid_te_flat = rt_e[-len(yte_seq):]

    print(f"    Stage3 RoR Strategies:")

    strategies = {}

    # Strategy A: Weighted LightGBM
    va_a, te_a, rmse_a, desc_a = ror_strategy_weighted_lgbm(
        ror_Xtr_c, ror_tr_c, ror_Xva, ror_Xte, s2_va, s2_te, y_va_arr, y_te_arr)
    te_rmse_a = np.sqrt(mean_squared_error(y_te_arr, te_a))
    strategies["A"] = {"va": rmse_a, "te": te_rmse_a, "desc": desc_a, "va_pred": va_a, "te_pred": te_a}
    print(f"      A. {desc_a}: Val={rmse_a:.4f} Test={te_rmse_a:.4f}")

    # Strategy B: Ridge
    va_b, te_b, rmse_b, desc_b = ror_strategy_ridge(
        ror_Xtr_c, ror_tr_c, ror_Xva, ror_Xte, s2_va, s2_te, y_va_arr, y_te_arr)
    te_rmse_b = np.sqrt(mean_squared_error(y_te_arr, te_b))
    strategies["B"] = {"va": rmse_b, "te": te_rmse_b, "desc": desc_b, "va_pred": va_b, "te_pred": te_b}
    print(f"      B. {desc_b}: Val={rmse_b:.4f} Test={te_rmse_b:.4f}")

    # Strategy C: ElasticNet
    va_c, te_c, rmse_c, desc_c = ror_strategy_elasticnet(
        ror_Xtr_c, ror_tr_c, ror_Xva, ror_Xte, s2_va, s2_te, y_va_arr, y_te_arr)
    te_rmse_c = np.sqrt(mean_squared_error(y_te_arr, te_c))
    strategies["C"] = {"va": rmse_c, "te": te_rmse_c, "desc": desc_c, "va_pred": va_c, "te_pred": te_c}
    print(f"      C. {desc_c}: Val={rmse_c:.4f} Test={te_rmse_c:.4f}")

    # Strategy D: OOF LightGBM
    va_d, te_d, rmse_d, desc_d = ror_strategy_oof_lgbm(
        ror_Xtr_c, ror_tr_c, ror_Xva, ror_Xte, s2_va, s2_te, y_va_arr, y_te_arr)
    te_rmse_d = np.sqrt(mean_squared_error(y_te_arr, te_d))
    strategies["D"] = {"va": rmse_d, "te": te_rmse_d, "desc": desc_d, "va_pred": va_d, "te_pred": te_d}
    print(f"      D. {desc_d}: Val={rmse_d:.4f} Test={te_rmse_d:.4f}")

    # Strategy E: Augmented LightGBM
    va_e, te_e, rmse_e, desc_e, _ = ror_strategy_augmented_lgbm(
        ror_Xtr_c, ror_tr_c, ror_Xva, ror_Xte, s2_va, s2_te, y_va_arr, y_te_arr,
        base_tr=base_tr_flat, base_va=base_va_flat, base_te=base_te_flat,
        resid_tr=resid_tr_flat, resid_va=resid_va_flat, resid_te=resid_te_flat)
    te_rmse_e = np.sqrt(mean_squared_error(y_te_arr, te_e))
    strategies["E"] = {"va": rmse_e, "te": te_rmse_e, "desc": desc_e, "va_pred": va_e, "te_pred": te_e}
    print(f"      E. {desc_e}: Val={rmse_e:.4f} Test={te_rmse_e:.4f}")

    # Strategy F: Ensemble
    va_f, te_f, rmse_f, desc_f = ror_strategy_ensemble(
        ror_Xtr_c, ror_tr_c, ror_Xva, ror_Xte, s2_va, s2_te, y_va_arr, y_te_arr)
    te_rmse_f = np.sqrt(mean_squared_error(y_te_arr, te_f))
    strategies["F"] = {"va": rmse_f, "te": te_rmse_f, "desc": desc_f, "va_pred": va_f, "te_pred": te_f}
    print(f"      F. {desc_f}: Val={rmse_f:.4f} Test={te_rmse_f:.4f}")

    # Strategy G: Random Walk Blend
    va_g, te_g, rmse_g, desc_g = ror_strategy_rw_blend(
        s2_va, s2_te, y_va_arr, y_te_arr, rw_va, rw_te)
    te_rmse_g = np.sqrt(mean_squared_error(y_te_arr, te_g))
    strategies["G"] = {"va": rmse_g, "te": te_rmse_g, "desc": desc_g, "va_pred": va_g, "te_pred": te_g}
    print(f"      G. {desc_g}: Val={rmse_g:.4f} Test={te_rmse_g:.4f}")

    # Strategy H: All-Feature LightGBM
    va_h, te_h, rmse_h, desc_h = ror_strategy_allfeature_lgbm(
        ror_Xtr_all, ror_tr_c, ror_Xva_all, ror_Xte_all, s2_va, s2_te, y_va_arr, y_te_arr)
    te_rmse_h = np.sqrt(mean_squared_error(y_te_arr, te_h))
    strategies["H"] = {"va": rmse_h, "te": te_rmse_h, "desc": desc_h, "va_pred": va_h, "te_pred": te_h}
    print(f"      H. {desc_h}: Val={rmse_h:.4f} Test={te_rmse_h:.4f}")

    # Pick best strategy by validation RMSE
    best_key = min(strategies, key=lambda k: strategies[k]["va"])
    best_strat = strategies[best_key]

    # Check if best RoR strategy beats S2 (no RoR)
    ror_improves = best_strat["va"] < s2_va_rmse

    if ror_improves:
        final_va = best_strat["va"]
        final_te = best_strat["te"]
        ror_used = f"Strategy_{best_key}"
        ror_desc = best_strat["desc"]
    else:
        final_va = s2_va_rmse
        final_te = s2_te_rmse
        ror_used = "None"
        ror_desc = "S2 baseline"

    elapsed = time.time() - t0
    print(f"    ---")
    print(f"    Best RoR: Strategy {best_key} ({best_strat['desc']})")
    print(f"    RoR vs S2: {best_strat['va']:.4f} vs {s2_va_rmse:.4f} -> {'IMPROVED' if ror_improves else 'NO IMPROVE'}")
    print(f"    * Final: Val={final_va:.4f} Test={final_te:.4f} ({elapsed:.0f}s)")

    results.append({
        "Experiment": label,
        "Base": bn, "Residual": rn,
        "Base_Val": b_va_rmse, "Base_Test": b_te_rmse,
        "S2_Val": s2_va_rmse, "S2_Test": s2_te_rmse,
        "RoR_Strategy": ror_used,
        "RoR_Desc": ror_desc,
        "RoR_Val": best_strat["va"], "RoR_Test": best_strat["te"],
        "RoR_Improved": ror_improves,
        "Final_Val": final_va, "Final_Test": final_te,
    })

    for k, v in strategies.items():
        all_ror_details.append({
            "Experiment": label,
            "Strategy": k,
            "Description": v["desc"],
            "Val_RMSE": v["va"],
            "Test_RMSE": v["te"],
            "Improves_S2": v["va"] < s2_va_rmse,
            "S2_Val": s2_va_rmse,
            "S2_Test": s2_te_rmse,
        })

# ── 8. RESULTS ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Phase 6: Results Summary")
print("=" * 70)

rdf = pd.DataFrame(results).sort_values("Final_Test").reset_index(drop=True)
ror_detail_df = pd.DataFrame(all_ror_details)

print(f"\n  {'Experiment':<35s} {'S2_Test':>8s} {'RoR_Test':>8s} {'Final':>8s} {'RoR':>12s}")
print("  " + "-" * 78)
for _, r in rdf.iterrows():
    print(f"  {r['Experiment']:<35s} {r['S2_Test']:>8.4f} {r['RoR_Test']:>8.4f} {r['Final_Test']:>8.4f} {r['RoR_Strategy']:>12s}")
print(f"  {'Random Walk':<35s} {'':>8s} {'':>8s} {rw_rmse:>8.4f}")

# Count RoR improvements
n_improved = sum(1 for r in results if r["RoR_Improved"])
print(f"\n  RoR Improvement Rate: {n_improved}/{len(results)} experiments")

# RoR detail summary
print(f"\n  RoR Strategy Breakdown (all experiments):")
for strat in ["A", "B", "C", "D", "E", "F", "G", "H"]:
    sdf = ror_detail_df[ror_detail_df["Strategy"] == strat]
    n_imp = sdf["Improves_S2"].sum()
    avg_va = sdf["Val_RMSE"].mean()
    print(f"    Strategy {strat}: Improves {n_imp}/{len(sdf)} | Avg Val RMSE: {avg_va:.4f}")

rdf.to_csv(f"{OUT}/experiment_results.csv", index=False)
ror_detail_df.to_csv(f"{OUT}/ror_strategy_details.csv", index=False)

# ── 9. VISUALIZATIONS ───────────────────────────────────────────────────
# Bar chart: All experiments
fig, ax = plt.subplots(figsize=(12, 7))
colors = ["#2ecc71" if r["RoR_Improved"] else "#3498db" for _, r in rdf.iterrows()]
bars = ax.barh(range(len(rdf)), rdf["Final_Test"].values, color=colors, alpha=0.85, label="Final (with RoR)")
# Overlay S2 as outline
ax.barh(range(len(rdf)), rdf["S2_Test"].values, color="none", edgecolor="#e74c3c", linewidth=1.5, linestyle="--", label="S2 (no RoR)")
ax.axvline(rw_rmse, color="red", ls="--", lw=2, label=f"Random Walk ({rw_rmse:.4f})")
ax.set_yticks(range(len(rdf)))
ax.set_yticklabels([f"{r['Experiment']} [{r['RoR_Strategy']}]" for _, r in rdf.iterrows()], fontsize=9)
ax.set_xlabel("Test RMSE"); ax.set_title("Enhanced RoR Experiments: Final vs S2")
ax.legend(fontsize=9); ax.invert_yaxis()
plt.tight_layout(); plt.savefig(f"{OUT}/03_all_experiments.png", dpi=150); plt.close()

# Stage progression for top 3
fig, axes = plt.subplots(1, min(3, len(rdf)), figsize=(15, 5))
if len(rdf) < 3:
    axes = [axes] if len(rdf) == 1 else list(axes)
for i in range(min(3, len(rdf))):
    r = rdf.iloc[i]
    ax = axes[i]
    bars = ax.bar(["Base", "+Resid(S2)", "+RoR(S3)"],
                  [r["Base_Test"], r["S2_Test"], r["RoR_Test"]],
                  color=["#3498db", "#2ecc71", "#e74c3c"], alpha=0.8)
    ax.axhline(rw_rmse, color="k", ls="--", label=f"RW({rw_rmse:.3f})")
    ax.set_ylabel("Test RMSE")
    ax.set_title(f"#{i+1}: {r['Base']}+{r['Residual']}\n{r['RoR_Desc']}", fontsize=9)
    ax.legend(fontsize=8)
    # Add value labels
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
plt.suptitle("Top 3: Stage Progression with Enhanced RoR", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/04_top3_stages.png", dpi=150); plt.close()

# RoR strategy comparison heatmap
pivot = ror_detail_df.pivot_table(index="Experiment", columns="Strategy", values="Val_RMSE")
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"Strategy {c}" for c in pivot.columns], fontsize=10)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, label="Val RMSE")
ax.set_title("RoR Strategy Comparison (Val RMSE)")
plt.tight_layout(); plt.savefig(f"{OUT}/05_ror_heatmap.png", dpi=150); plt.close()

# SHAP plot
fig, ax = plt.subplots(figsize=(10, 8))
t20 = shap_df.head(20)
ax.barh(range(len(t20)), t20["shap"].values[::-1])
ax.set_yticks(range(len(t20)))
ax.set_yticklabels(t20["feature"].values[::-1], fontsize=8)
ax.set_xlabel("Mean |SHAP|"); ax.set_title("SHAP Top 20 Features")
plt.tight_layout(); plt.savefig(f"{OUT}/01_shap.png", dpi=150); plt.close()

# CV curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cv_df["n"], cv_df["cv"], "bo-")
ax.axvline(best_n, color="r", ls="--", label=f"best={best_n}")
ax.set_xlabel("# Features"); ax.set_ylabel("CV RMSE")
ax.set_title("Feature Selection CV"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/02_cv_curve.png", dpi=150); plt.close()

# Config
cfg = {
    "target": TARGET,
    "n_experiments": len(EXPS),
    "feature_selection": {"method": "SHAP+TimeSeriesCV", "selected": best_n, "total": len(ALL_FEAT)},
    "models": list(MODELS.keys()),
    "seq_len": SEQ_LEN,
    "n_seeds": N_SEEDS,
    "ror_strategies": ["A:WeightedLGBM", "B:Ridge", "C:ElasticNet", "D:OOF_LGBM", "E:AugmentedLGBM", "F:Ensemble", "G:RW_Blend", "H:AllFeatureLGBM"],
    "best": rdf.iloc[0]["Experiment"],
    "best_rmse": float(rdf.iloc[0]["Final_Test"]),
    "best_ror_strategy": rdf.iloc[0]["RoR_Strategy"],
    "best_ror_desc": rdf.iloc[0]["RoR_Desc"],
    "ror_improved": bool(rdf.iloc[0]["RoR_Improved"]),
    "n_ror_improved": int(n_improved),
    "rw_rmse": float(rw_rmse),
}
json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=2, default=str)

print(f"\n  Best: {rdf.iloc[0]['Experiment']} ({rdf.iloc[0]['RoR_Desc']})")
print(f"       Test RMSE={rdf.iloc[0]['Final_Test']:.4f}")
print(f"  Random Walk: {rw_rmse:.4f}")
print(f"  RoR Success: {n_improved}/{len(results)} experiments improved by RoR")
print(f"  Output: {OUT}/")
print("=" * 70)
