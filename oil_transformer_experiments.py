"""
Brent Oil Forecasting: SHAP Feature Selection + 10 Transformer Experiments
Pipeline: Baseline → Residual → RoR (LightGBM)
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
OUT = "output_oil_transformer"
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
print(f"  ★ Optimal: {best_n} features (CV={cv_df['cv'].min():.4f})")

shap_df.to_csv(f"{OUT}/shap_importance.csv", index=False)

# SHAP plot
fig, ax = plt.subplots(figsize=(10, 8))
t20 = shap_df.head(20)
ax.barh(range(len(t20)), t20["shap"].values[::-1])
ax.set_yticks(range(len(t20)))
ax.set_yticklabels(t20["feature"].values[::-1], fontsize=8)
ax.set_xlabel("Mean |SHAP|"); ax.set_title("SHAP Top 20")
plt.tight_layout(); plt.savefig(f"{OUT}/01_shap.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cv_df["n"], cv_df["cv"], "bo-")
ax.axvline(best_n, color="r", ls="--", label=f"best={best_n}")
ax.set_xlabel("# Features"); ax.set_ylabel("CV RMSE")
ax.set_title("Feature Selection CV"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/02_cv_curve.png", dpi=150); plt.close()

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
# For val: use last SEQ_LEN of train as buffer
buf_X = Xtr_sc[-SEQ_LEN:]
buf_y = ytr_n[-SEQ_LEN:]
Xva_seq, yva_seq = mkseq(np.vstack([buf_X, Xva_sc]),
                          np.concatenate([buf_y, yva_n]), SEQ_LEN)
# For test: use last SEQ_LEN of (train+val) as buffer
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
print(f"  Models: {list(MODELS.keys())}")

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

# ── 6. 10 EXPERIMENTS ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Phase 5: 10 Transformer Experiments")
print("=" * 70)

EXPS = [
    ("NLinear",      "NLinear"),
    ("PatchTST",     "PatchTST"),
    ("iTransformer", "iTransformer"),
    ("Transformer",  "Transformer"),
    ("LSTM",         "LSTM"),
    ("PatchTST",     "NLinear"),
    ("iTransformer", "NLinear"),
    ("NLinear",      "PatchTST"),
    ("PatchTST",     "iTransformer"),
    ("Transformer",  "NLinear"),
]

# Random Walk
rw_va = np.array([y.iloc[y.index.get_loc(i)-1] for i in y_va.index])
rw_te = np.array([y.iloc[y.index.get_loc(i)-1] for i in y_te.index])
rw_rmse = np.sqrt(mean_squared_error(y_te, rw_te))
print(f"  Random Walk: Test RMSE={rw_rmse:.4f}")

N_SEEDS = 3
results = []

for ei, (bn, rn) in enumerate(EXPS):
    label = f"Exp{ei+1}: {bn}+{rn}+LGBM"
    t0 = time.time()
    print(f"\n  {'─'*60}")
    print(f"  {label}")

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
    print(f"    Base({bn}): Val={b_va_rmse:.4f} Test={b_te_rmse:.4f}")

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
    print(f"    +Resid({rn}): Val={s2_va_rmse:.4f} Test={s2_te_rmse:.4f}")

    # Stage 3: RoR (LightGBM)
    ror_tr = resid_tr_tgt - rtr_e
    ror_Xtr = Xtr_sc[SEQ_LEN:][:len(ror_tr)]
    ror_Xva = Xva_sc[-len(yva_seq):]
    ror_Xte = Xte_sc[-len(yte_seq):]
    mn = min(len(ror_tr), len(ror_Xtr))
    ror_tr, ror_Xtr = ror_tr[:mn], ror_Xtr[:mn]

    lgb_r = lgb.LGBMRegressor(learning_rate=0.02, num_leaves=10,
        min_child_samples=40, subsample=0.6, colsample_bytree=0.5,
        reg_alpha=2.0, reg_lambda=2.0, n_estimators=300, verbose=-1, random_state=SEED)
    lgb_r.fit(ror_Xtr, ror_tr)

    ror_va = lgb_r.predict(ror_Xva)
    ror_te = lgb_r.predict(ror_Xte)
    s3_va = s2_va + ror_va * ystd
    s3_te = s2_te + ror_te * ystd

    s3_va_rmse = np.sqrt(mean_squared_error(y_va.values[-len(s3_va):], s3_va))
    s3_te_rmse = np.sqrt(mean_squared_error(y_te.values[-len(s3_te):], s3_te))

    # Validation gate
    if s3_va_rmse < s2_va_rmse:
        use_ror = True; f_va, f_te = s3_va_rmse, s3_te_rmse
    else:
        use_ror = False; f_va, f_te = s2_va_rmse, s2_te_rmse

    elapsed = time.time() - t0
    print(f"    +RoR: Val={s3_va_rmse:.4f} Test={s3_te_rmse:.4f} Gate={'PASS' if use_ror else 'BLOCKED'}")
    print(f"    ★ Final: Val={f_va:.4f} Test={f_te:.4f} ({elapsed:.0f}s)")

    results.append({
        "Experiment": label, "Base": bn, "Residual": rn,
        "Base_Val": b_va_rmse, "Base_Test": b_te_rmse,
        "S2_Val": s2_va_rmse, "S2_Test": s2_te_rmse,
        "S3_Val": s3_va_rmse, "S3_Test": s3_te_rmse,
        "RoR": use_ror, "Final_Val": f_va, "Final_Test": f_te,
    })

# ── 7. RESULTS ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Phase 6: Results Summary")
print("=" * 70)

rdf = pd.DataFrame(results).sort_values("Final_Test").reset_index(drop=True)

print(f"\n  {'Experiment':<42s} {'Val':>8s} {'Test':>8s} {'RoR':>5s}")
print("  " + "─" * 68)
for _, r in rdf.iterrows():
    print(f"  {r['Experiment']:<42s} {r['Final_Val']:>8.4f} {r['Final_Test']:>8.4f} {'Y' if r['RoR'] else 'N':>5s}")
print(f"  {'Random Walk':<42s} {np.sqrt(mean_squared_error(y_va, rw_va)):>8.4f} {rw_rmse:>8.4f}")

print("\n  ★ TOP 3:")
for i, r in rdf.head(3).iterrows():
    print(f"    #{i+1} {r['Experiment']}")
    print(f"       Base={r['Base_Test']:.4f} → +Resid={r['S2_Test']:.4f} → +RoR={r['S3_Test']:.4f} → Final={r['Final_Test']:.4f}")

rdf.to_csv(f"{OUT}/experiment_results.csv", index=False)

# Visualization
fig, ax = plt.subplots(figsize=(12, 7))
colors = ["#2ecc71" if r["Final_Test"] <= rw_rmse else "#3498db" for _, r in rdf.iterrows()]
ax.barh(range(len(rdf)), rdf["Final_Test"].values, color=colors, alpha=0.85)
ax.axvline(rw_rmse, color="red", ls="--", lw=2, label=f"Random Walk ({rw_rmse:.4f})")
ax.set_yticks(range(len(rdf)))
ax.set_yticklabels([r["Experiment"] for _, r in rdf.iterrows()], fontsize=9)
ax.set_xlabel("Test RMSE"); ax.set_title("10 Transformer Experiments")
ax.legend(); ax.invert_yaxis()
plt.tight_layout(); plt.savefig(f"{OUT}/03_all_experiments.png", dpi=150); plt.close()

# Top 3 stage progression
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (_, r) in enumerate(rdf.head(3).iterrows()):
    ax = axes[i]
    ax.bar(["Base","+Resid","+RoR"], [r["Base_Test"],r["S2_Test"],r["S3_Test"]],
           color=["#3498db","#2ecc71","#e74c3c"], alpha=0.8)
    ax.axhline(rw_rmse, color="k", ls="--", label="RW")
    ax.set_ylabel("Test RMSE"); ax.set_title(f"#{i+1}: {r['Base']}+{r['Residual']}", fontsize=10)
    ax.legend(fontsize=8)
plt.suptitle("Top 3 Stage Progression", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/04_top3.png", dpi=150); plt.close()

# Config
cfg = {
    "target": TARGET, "n_experiments": 10,
    "feature_selection": {"method": "SHAP+TimeSeriesCV", "selected": best_n, "total": len(ALL_FEAT)},
    "models": list(MODELS.keys()), "seq_len": SEQ_LEN, "n_seeds": N_SEEDS,
    "best": rdf.iloc[0]["Experiment"], "best_rmse": float(rdf.iloc[0]["Final_Test"]),
    "rw_rmse": float(rw_rmse),
}
json.dump(cfg, open(f"{OUT}/config.json","w"), indent=2, default=str)

print(f"\n  Best: {rdf.iloc[0]['Experiment']} — Test RMSE={rdf.iloc[0]['Final_Test']:.4f}")
print(f"  Random Walk: {rw_rmse:.4f}")
print(f"  Output: {OUT}/")
print("=" * 70)
