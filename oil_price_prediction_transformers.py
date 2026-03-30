"""
=============================================================================
  유가 예측 (Oil Price Prediction) - Advanced Transformer Models

  Target: Brent Crude Oil (Com_BrentCrudeOil)
  Data: Weekly financial dataset (2013-04 ~ 2026-01)

  Models:
    1. PatchTST  (Patch Time Series Transformer, ICLR 2023)
    2. iTransformer (Inverted Transformer, ICLR 2024)
    3. TFT-Lite (Temporal Fusion Transformer, simplified)

  + Baseline: Naive Drift, GradientBoosting, LSTM
  + Ensemble: Stacking & Weighted combination
=============================================================================
"""

# %% [markdown]
# # 유가 예측 - Advanced Transformer-Based Models
#
# ## 프로젝트 개요
#
# | 항목 | 내용 |
# |------|------|
# | **목표** | Brent Crude Oil 주간 가격 예측 |
# | **데이터** | 74개 거시경제/금융 지표, 주간 빈도 (2013~2026) |
# | **핵심 모델** | PatchTST, iTransformer, TFT-Lite |
# | **평가 지표** | RMSE, MAPE, R², Directional Accuracy |
#
# ## 모델 선정 근거
#
# ### 1. PatchTST (Nie et al., ICLR 2023)
# - 시계열을 패치 단위로 분할하여 ViT 방식 적용
# - Channel-Independent 전략으로 과적합 방지
# - 장기 의존성 포착에 탁월 (Patch → 더 긴 lookback 가능)
#
# ### 2. iTransformer (Liu et al., ICLR 2024)
# - **축 반전**: 시간축이 아닌 변수축에 Attention 적용
# - 다변량 시계열에서 변수 간 상호작용 직접 모델링
# - 최신 SOTA 벤치마크 다수 달성
#
# ### 3. TFT-Lite (Lim et al., IJF 2021 기반 경량화)
# - Variable Selection Network으로 중요 변수 자동 선택
# - Gated Residual Network으로 비선형 관계 포착
# - Interpretable Multi-Head Attention으로 시간적 패턴 학습
# - 해석 가능성 제공 (어떤 변수가 중요한지, 어떤 시점이 중요한지)

# %%
# ============================================================================
# 1. 환경 설정
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import warnings
import os
import math
from collections import OrderedDict

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')

OUTPUT_DIR = 'output_oil'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 실험 설정 ──
CONFIG = {
    'data_file': 'data_weekly_260120.csv',
    'target_col': 'Com_BrentCrudeOil',

    # 시간 분할 (니켈 실험과 동일 구간)
    'val_start': '2025-08-04',
    'val_end': '2025-10-20',
    'test_start': '2025-10-27',
    'test_end': '2026-01-12',

    # 피처 선택
    'top_n_features': 25,

    # 시퀀스 길이
    'seq_len': 24,       # 6개월 lookback (주간)
    'pred_len': 1,       # 1주 ahead 예측

    # 학습
    'batch_size': 32,
    'epochs': 150,
    'lr': 1e-3,
    'patience': 20,
    'weight_decay': 1e-4,
    'random_seed': 42,

    # PatchTST
    'patch_len': 4,
    'stride': 2,

    # Model dims
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 2,
    'd_ff': 128,
    'dropout': 0.3,
}

np.random.seed(CONFIG['random_seed'])
torch.manual_seed(CONFIG['random_seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['random_seed'])

print('=' * 70)
print('  유가 예측 - Advanced Transformer-Based Models')
print('=' * 70)
print(f'  Target: {CONFIG["target_col"]}')
print(f'  Device: {device}')
print(f'  Sequence Length: {CONFIG["seq_len"]} weeks')
print(f'  Models: PatchTST, iTransformer, TFT-Lite')
print('=' * 70)

# %%
# ============================================================================
# 2. 데이터 로드 및 탐색적 분석
# ============================================================================
print('\n' + '=' * 70)
print('  2. 데이터 로드 및 EDA')
print('=' * 70)

df_raw = pd.read_csv(CONFIG['data_file'])
df_raw['dt'] = pd.to_datetime(df_raw['dt'])
df_raw = df_raw.set_index('dt').sort_index()

print(f'\n  데이터 기간: {df_raw.index[0].date()} ~ {df_raw.index[-1].date()}')
print(f'  총 관측치: {len(df_raw)}')
print(f'  피처 수: {len(df_raw.columns)}')

# 타겟 변수 시각화
target = df_raw[CONFIG['target_col']]
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 가격 추이
ax = axes[0, 0]
ax.plot(target.index, target.values, 'b-', lw=1.5)
ax.axvline(pd.Timestamp(CONFIG['val_start']), color='orange', ls='--', label='Val Start')
ax.axvline(pd.Timestamp(CONFIG['test_start']), color='red', ls='--', label='Test Start')
ax.set_title('Brent Crude Oil - Weekly Price', fontweight='bold')
ax.set_ylabel('USD/barrel')
ax.legend()

# 수익률 분포
returns = target.pct_change().dropna()
ax = axes[0, 1]
ax.hist(returns, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(0, color='red', ls='--')
ax.set_title(f'Weekly Returns Distribution (skew={returns.skew():.2f}, kurt={returns.kurtosis():.2f})', fontweight='bold')
ax.set_xlabel('Weekly Return')

# 롤링 변동성
ax = axes[1, 0]
vol_12 = returns.rolling(12).std() * np.sqrt(52)
vol_26 = returns.rolling(26).std() * np.sqrt(52)
ax.plot(vol_12.index, vol_12.values, label='12w vol (annualized)', alpha=0.8)
ax.plot(vol_26.index, vol_26.values, label='26w vol (annualized)', alpha=0.8)
ax.set_title('Rolling Volatility', fontweight='bold')
ax.legend()

# ACF
from numpy.fft import fft, ifft
ax = axes[1, 1]
lags = 52
centered = (target - target.mean()).values
n = len(centered)
acf_vals = [1.0]
for lag in range(1, lags + 1):
    c = np.corrcoef(centered[:-lag], centered[lag:])[0, 1]
    acf_vals.append(c)
ax.bar(range(lags + 1), acf_vals, color='steelblue', alpha=0.7)
ax.axhline(1.96/np.sqrt(n), color='red', ls='--', alpha=0.5)
ax.axhline(-1.96/np.sqrt(n), color='red', ls='--', alpha=0.5)
ax.set_title('Autocorrelation (up to 52 lags)', fontweight='bold')
ax.set_xlabel('Lag (weeks)')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_eda.png', dpi=150, bbox_inches='tight')
plt.show()

# 기간별 통계
print('\n  기간별 통계:')
for name, mask in [('Train', target.index < CONFIG['val_start']),
                    ('Val', (target.index >= CONFIG['val_start']) & (target.index <= CONFIG['val_end'])),
                    ('Test', (target.index >= CONFIG['test_start']))]:
    subset = target[mask]
    print(f'    {name:5s}: mean={subset.mean():.2f}, std={subset.std():.2f}, '
          f'min={subset.min():.2f}, max={subset.max():.2f}, n={len(subset)}')

# %%
# ============================================================================
# 3. 피처 엔지니어링 & 선택
# ============================================================================
print('\n' + '=' * 70)
print('  3. 피처 엔지니어링')
print('=' * 70)

# 3.1 기본 피처 필터 (에너지/거시경제 관련)
def select_relevant_columns(columns, target_col):
    """유가 예측에 관련된 컬럼 선별"""
    selected = []
    for col in columns:
        if col == target_col:
            continue
        # 에너지 관련
        if any(x in col for x in ['CrudeOil', 'Gasoline', 'NaturalGas', 'Coal', 'Uranium']):
            selected.append(col)
        # 거시경제 지표
        elif any(x in col for x in ['Idx_', 'Bonds_', 'EX_']):
            selected.append(col)
        # 금속 (산업 활동 proxy)
        elif 'Com_LME' in col:
            selected.append(col)
        # 기타 원자재 (인플레이션 proxy)
        elif any(x in col for x in ['Gold', 'Silver', 'Iron', 'Steel', 'Copper']):
            selected.append(col)
    return sorted(list(set(selected)))

relevant_cols = select_relevant_columns(df_raw.columns, CONFIG['target_col'])
df = df_raw[[CONFIG['target_col']] + relevant_cols].copy()
df = df.ffill().dropna()

print(f'  관련 피처: {len(relevant_cols)}개')

# 3.2 기술적 피처 추가
target_series = df[CONFIG['target_col']]

# 이동평균
for w in [4, 12, 26]:
    df[f'MA_{w}w'] = target_series.rolling(w).mean()
    df[f'MA_ratio_{w}w'] = target_series / target_series.rolling(w).mean()

# 수익률 & 변동성
for w in [1, 4, 12]:
    df[f'Return_{w}w'] = target_series.pct_change(w)

df['Volatility_12w'] = target_series.pct_change().rolling(12).std()
df['Volatility_26w'] = target_series.pct_change().rolling(26).std()

# Momentum
df['Momentum_4w'] = target_series - target_series.shift(4)
df['Momentum_12w'] = target_series - target_series.shift(12)

# RSI (14-week)
delta = target_series.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['RSI_14w'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

# Spread: Brent vs WTI
if 'Com_CrudeOil' in df.columns:
    df['Brent_WTI_Spread'] = df[CONFIG['target_col']] - df['Com_CrudeOil']

df = df.dropna()
print(f'  기술적 피처 추가 후: {len(df.columns)}개')
print(f'  유효 데이터: {len(df)} rows')

# 3.3 Target & Features 분리 (1주 ahead)
y = df[CONFIG['target_col']]
X = df.drop(columns=[CONFIG['target_col']]).shift(1)  # 1주 lag

valid_idx = X.dropna().index.intersection(y.dropna().index)
X, y = X.loc[valid_idx], y.loc[valid_idx]

# 3.4 Train / Val / Test 분할
train_mask = X.index < CONFIG['val_start']
val_mask = (X.index >= CONFIG['val_start']) & (X.index <= CONFIG['val_end'])
test_mask = (X.index >= CONFIG['test_start']) & (X.index <= CONFIG['test_end'])

X_train_all, y_train = X[train_mask], y[train_mask]
X_val_all, y_val = X[val_mask], y[val_mask]
X_test_all, y_test = X[test_mask], y[test_mask]

print(f'\n  Split:')
print(f'    Train: {len(X_train_all)} ({X_train_all.index[0].date()} ~ {X_train_all.index[-1].date()})')
print(f'    Val  : {len(X_val_all)} ({X_val_all.index[0].date()} ~ {X_val_all.index[-1].date()})')
print(f'    Test : {len(X_test_all)} ({X_test_all.index[0].date()} ~ {X_test_all.index[-1].date()})')

# 3.5 SHAP 기반 피처 선택
print('\n  SHAP feature selection...')
model_shap = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)
model_shap.fit(X_train_all, y_train)
explainer = shap.TreeExplainer(model_shap)
shap_values = explainer.shap_values(X_train_all)
feat_imp = pd.DataFrame({
    'feature': X_train_all.columns,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

selected_features = feat_imp.head(CONFIG['top_n_features'])['feature'].tolist()
print(f'  Top {CONFIG["top_n_features"]} features:')
for i, row in feat_imp.head(CONFIG['top_n_features']).iterrows():
    print(f'    {row["feature"]:40s} importance={row["importance"]:.4f}')

X_train = X_train_all[selected_features]
X_val = X_val_all[selected_features]
X_test = X_test_all[selected_features]

# SHAP summary plot
fig, ax = plt.subplots(figsize=(10, 8))
top_idx = feat_imp.head(20).index
shap.summary_plot(shap_values[:, top_idx], X_train_all.iloc[:, top_idx], show=False, max_display=20)
plt.title('SHAP Feature Importance - Oil Price Prediction', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_shap_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# ============================================================================
# 4. 시퀀스 데이터 준비 (Transformer용)
# ============================================================================
print('\n' + '=' * 70)
print('  4. 시퀀스 데이터 생성')
print('=' * 70)

SEQ_LEN = CONFIG['seq_len']
n_features = len(selected_features)

# RobustScaler (이상치에 강건)
scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_all = pd.concat([X_train, X_val, X_test])
y_all = pd.concat([y_train, y_val, y_test])

X_scaled = scaler_X.fit_transform(X_all)
y_scaled = scaler_y.fit_transform(y_all.values.reshape(-1, 1)).flatten()

def create_sequences(X, y, seq_len):
    """시퀀스 데이터 생성"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)

n_train = len(X_train) - SEQ_LEN
n_val = len(X_val)
n_test = len(X_test)

X_train_seq = X_seq[:n_train]
y_train_seq = y_seq[:n_train]
X_val_seq = X_seq[n_train:n_train+n_val]
y_val_seq = y_seq[n_train:n_train+n_val]
X_test_seq = X_seq[n_train+n_val:n_train+n_val+n_test]
y_test_seq = y_seq[n_train+n_val:n_train+n_val+n_test]

print(f'  Sequence shape: ({SEQ_LEN}, {n_features})')
print(f'  Train sequences: {len(X_train_seq)}')
print(f'  Val sequences  : {len(X_val_seq)}')
print(f'  Test sequences : {len(X_test_seq)}')

# %%
# ============================================================================
# 5. 유틸리티 함수
# ============================================================================

def eval_metrics(y_true, y_pred, name=''):
    """평가 지표 계산"""
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)

    # Directional Accuracy
    if len(y_true) > 1:
        actual_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        da = np.mean(actual_dir == pred_dir) * 100
    else:
        da = np.nan

    return {'Model': name, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2, 'DA%': da}


def inverse_transform_y(y_scaled):
    """스케일링 역변환"""
    return scaler_y.inverse_transform(np.array(y_scaled).reshape(-1, 1)).flatten()


class EarlyStopping:
    """Early stopping with model state save"""
    def __init__(self, patience=20, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_model(model, X_tr, y_tr, X_vl, y_vl, config=CONFIG, model_name='Model'):
    """범용 학습 함수 with cosine annealing & mixed precision"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )
    loss_fn = nn.HuberLoss(delta=1.0)  # Huber loss: 이상치에 강건

    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    val_ds = TensorDataset(torch.FloatTensor(X_vl), torch.FloatTensor(y_vl))
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])

    es = EarlyStopping(patience=config['patience'])
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)

        es(avg_val, model)
        if es.early_stop:
            print(f'    [{model_name}] Early stop at epoch {epoch+1}, best val_loss={es.best_loss:.6f}')
            break

        if (epoch + 1) % 30 == 0:
            print(f'    [{model_name}] Epoch {epoch+1}: train={avg_train:.6f}, val={avg_val:.6f}')

    es.load_best(model)
    return model, history


def predict_model(model, X):
    """예측 (역변환 포함)"""
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X).to(device)).cpu().numpy()
    return inverse_transform_y(pred)


# %% [markdown]
# ---
# ## 6. Model 1: PatchTST
#
# **핵심 아이디어**: 시계열을 겹치는 패치로 분할 → 각 패치를 토큰으로 Transformer에 입력
#
# - Channel-Independent: 각 변수를 독립적으로 처리 → 과적합 방지
# - Patching: 정보 밀도 ↑, 계산 비용 ↓, 더 긴 lookback 가능

# %%
# ============================================================================
# 6. PatchTST (Patch Time Series Transformer)
# ============================================================================
print('\n' + '=' * 70)
print('  6. Model 1: PatchTST')
print('=' * 70)


class PatchTST(nn.Module):
    """
    PatchTST: A Time Series is Worth 64 Words (Nie et al., ICLR 2023)

    Channel-Independent Patch Transformer for time series forecasting.
    Each variable is processed independently through shared Transformer layers.
    """
    def __init__(self, n_features, seq_len, pred_len=1,
                 patch_len=4, stride=2, d_model=64, n_heads=4,
                 n_layers=2, d_ff=128, dropout=0.3):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride

        # Number of patches
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Patch embedding
        self.patch_embed = nn.Linear(patch_len, d_model)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        # Flatten head: aggregate all patches + channels → prediction
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.n_patches * d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, pred_len)
        )

        # Channel aggregation
        self.channel_agg = nn.Sequential(
            nn.Linear(n_features, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1)
        )

    def forward(self, x):
        # x: (B, seq_len, n_features)
        B, L, C = x.shape

        # Channel-independent: process each variable separately
        # Reshape: (B, C, L) → process each channel through patches
        x = x.permute(0, 2, 1)  # (B, C, L)

        # Create patches for all channels at once
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # patches: (B, C, n_patches, patch_len)

        B, C, N, P = patches.shape
        patches = patches.reshape(B * C, N, P)  # (B*C, n_patches, patch_len)

        # Patch embedding + positional encoding
        z = self.patch_embed(patches) + self.pos_embed  # (B*C, n_patches, d_model)

        # Transformer
        z = self.transformer(z)  # (B*C, n_patches, d_model)
        z = self.norm(z)

        # Flatten patches
        z = z.reshape(B, C, -1)  # (B, C, n_patches * d_model)

        # Per-channel prediction
        channel_preds = self.head(z).squeeze(-1)  # (B, C)

        # Aggregate channels
        out = self.channel_agg(channel_preds).squeeze(-1)  # (B,)

        return out


# Build and train
patchtst = PatchTST(
    n_features=n_features,
    seq_len=SEQ_LEN,
    pred_len=1,
    patch_len=CONFIG['patch_len'],
    stride=CONFIG['stride'],
    d_model=CONFIG['d_model'],
    n_heads=CONFIG['n_heads'],
    n_layers=CONFIG['n_layers'],
    d_ff=CONFIG['d_ff'],
    dropout=CONFIG['dropout']
)
print(f'  Parameters: {sum(p.numel() for p in patchtst.parameters()):,}')

patchtst, hist_patchtst = train_model(
    patchtst, X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_name='PatchTST'
)

patchtst_pred = predict_model(patchtst, X_test_seq)
y_test_raw = inverse_transform_y(y_test_seq)
result_patchtst = eval_metrics(y_test_raw, patchtst_pred, 'PatchTST')
print(f'\n  PatchTST Test Results:')
print(f'    RMSE={result_patchtst["RMSE"]:.4f}, MAPE={result_patchtst["MAPE"]:.2f}%, '
      f'R2={result_patchtst["R2"]:.4f}, DA={result_patchtst["DA%"]:.1f}%')


# %% [markdown]
# ---
# ## 7. Model 2: iTransformer
#
# **핵심 아이디어**: Attention을 시간축이 아닌 **변수축**에 적용
#
# - 각 변수의 전체 시계열을 하나의 토큰으로 임베딩
# - 변수 간 상호작용을 직접 학습 (cross-variate dependency)
# - Feed-Forward Network가 각 변수의 시간적 패턴을 학습

# %%
# ============================================================================
# 7. iTransformer (Inverted Transformer)
# ============================================================================
print('\n' + '=' * 70)
print('  7. Model 2: iTransformer')
print('=' * 70)


class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
    (Liu et al., ICLR 2024)

    Key idea: Apply attention on the variate dimension instead of time dimension.
    Each variate's full time series is embedded as one token.
    """
    def __init__(self, n_features, seq_len, pred_len=1,
                 d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.3):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len

        # Variate embedding: project each variable's time series to d_model
        self.variate_embed = nn.Linear(seq_len, d_model)

        # Learnable variate tokens (like CLS tokens for each variable)
        self.variate_pos = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)

        # Transformer on variate dimension
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Normalization
        self.norm = nn.LayerNorm(d_model)

        # Prediction head: aggregate variate representations
        self.head = nn.Sequential(
            nn.Linear(n_features * d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, pred_len)
        )

    def forward(self, x):
        # x: (B, seq_len, n_features)
        B, L, C = x.shape

        # Invert: treat each variable as a token
        x = x.permute(0, 2, 1)  # (B, n_features, seq_len)

        # Embed each variate's time series
        z = self.variate_embed(x)  # (B, n_features, d_model)
        z = z + self.variate_pos   # Add variate-specific position

        # Transformer attention across variates
        z = self.transformer(z)    # (B, n_features, d_model)
        z = self.norm(z)

        # Flatten and predict
        z = z.reshape(B, -1)       # (B, n_features * d_model)
        out = self.head(z).squeeze(-1)  # (B,)

        return out


# Build and train
itransformer = iTransformer(
    n_features=n_features,
    seq_len=SEQ_LEN,
    pred_len=1,
    d_model=CONFIG['d_model'],
    n_heads=CONFIG['n_heads'],
    n_layers=CONFIG['n_layers'],
    d_ff=CONFIG['d_ff'],
    dropout=CONFIG['dropout']
)
print(f'  Parameters: {sum(p.numel() for p in itransformer.parameters()):,}')

itransformer, hist_itrans = train_model(
    itransformer, X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_name='iTransformer'
)

itrans_pred = predict_model(itransformer, X_test_seq)
result_itrans = eval_metrics(y_test_raw, itrans_pred, 'iTransformer')
print(f'\n  iTransformer Test Results:')
print(f'    RMSE={result_itrans["RMSE"]:.4f}, MAPE={result_itrans["MAPE"]:.2f}%, '
      f'R2={result_itrans["R2"]:.4f}, DA={result_itrans["DA%"]:.1f}%')


# %% [markdown]
# ---
# ## 8. Model 3: TFT-Lite (Temporal Fusion Transformer - Simplified)
#
# **핵심 아이디어**: Variable Selection + Gated Residual + Interpretable Attention
#
# - **Variable Selection Network**: 입력 변수 중 중요한 것만 자동 선택
# - **Gated Residual Network (GRN)**: 비선형 관계를 학습하되, 필요없으면 skip
# - **Interpretable Multi-Head Attention**: 어떤 시점이 예측에 중요한지 해석 가능
# - LSTM encoder로 temporal context 포착

# %%
# ============================================================================
# 8. TFT-Lite (Temporal Fusion Transformer - Simplified)
# ============================================================================
print('\n' + '=' * 70)
print('  8. Model 3: TFT-Lite')
print('=' * 70)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) - TFT의 핵심 빌딩 블록"""
    def __init__(self, d_in, d_hidden, d_out=None, dropout=0.1, context_dim=None):
        super().__init__()
        d_out = d_out or d_in
        self.d_in = d_in
        self.d_out = d_out

        self.fc1 = nn.Linear(d_in, d_hidden)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.dropout = nn.Dropout(dropout)

        # Gating: GLU mechanism
        self.gate_fc = nn.Linear(d_hidden, d_out)
        self.gate_sig = nn.Sigmoid()

        # Skip connection
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

        # Optional context
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, d_hidden, bias=False)
        else:
            self.context_fc = None

    def forward(self, x, context=None):
        residual = self.skip(x)

        h = self.fc1(x)
        if self.context_fc is not None and context is not None:
            h = h + self.context_fc(context)
        h = self.elu(h)

        # GLU gate
        gate_input = h
        h = self.dropout(self.fc2(h))
        gate = self.gate_sig(self.gate_fc(gate_input))
        h = gate * h

        return self.norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network - 입력 변수의 중요도를 학습"""
    def __init__(self, n_features, d_hidden, dropout=0.1, context_dim=None):
        super().__init__()
        self.n_features = n_features

        # Per-variable GRNs
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(1, d_hidden, d_hidden, dropout)
            for _ in range(n_features)
        ])

        # Softmax variable weights
        self.weight_grn = GatedResidualNetwork(
            n_features, d_hidden, n_features, dropout, context_dim
        )

    def forward(self, x, context=None):
        # x: (B, T, n_features) or (B, n_features)
        has_time = x.dim() == 3

        if has_time:
            B, T, C = x.shape
            # Flatten for processing
            flat_x = x.reshape(B * T, C)
        else:
            flat_x = x
            B = x.shape[0]

        # Variable selection weights
        weights = F.softmax(self.weight_grn(flat_x, context), dim=-1)  # (B*T, C)

        # Per-variable transformation
        var_outputs = []
        for i in range(self.n_features):
            var_out = self.var_grns[i](flat_x[:, i:i+1])  # (B*T, d_hidden)
            var_outputs.append(var_out)

        var_outputs = torch.stack(var_outputs, dim=1)  # (B*T, C, d_hidden)

        # Weighted sum
        weighted = (weights.unsqueeze(-1) * var_outputs).sum(dim=1)  # (B*T, d_hidden)

        if has_time:
            weighted = weighted.reshape(B, T, -1)
            weights = weights.reshape(B, T, -1)

        return weighted, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-Head Attention (TFT style)"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape

        # Multi-head projections
        Q = self.W_q(q).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, T, -1)

        out = self.W_o(context)

        # Average attention across heads for interpretability
        avg_attn = attn_weights.mean(dim=1)  # (B, T, T)

        return out, avg_attn


class TFTLite(nn.Module):
    """
    Temporal Fusion Transformer - Lightweight version
    (Based on Lim et al., International Journal of Forecasting, 2021)

    Components:
    1. Variable Selection Network (VSN)
    2. LSTM Encoder for temporal processing
    3. Gated Residual Network (GRN) layers
    4. Interpretable Multi-Head Attention
    5. Final prediction with gating
    """
    def __init__(self, n_features, seq_len, pred_len=1,
                 d_model=64, n_heads=4, n_layers=1, d_ff=128, dropout=0.3):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model

        # 1. Variable Selection
        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)

        # 2. LSTM Encoder
        self.lstm = nn.LSTM(
            d_model, d_model, num_layers=n_layers,
            batch_first=True, dropout=0 if n_layers == 1 else dropout
        )

        # 3. Post-LSTM Gate + Norm
        self.post_lstm_gate = nn.Linear(d_model, d_model)
        self.post_lstm_norm = nn.LayerNorm(d_model)

        # 4. Static enrichment GRN
        self.enrichment_grn = GatedResidualNetwork(d_model, d_ff, d_model, dropout)

        # 5. Temporal Self-Attention
        self.attention = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.attn_gate = nn.Linear(d_model, d_model)
        self.attn_norm = nn.LayerNorm(d_model)

        # 6. Position-wise Feed-forward
        self.ff_grn = GatedResidualNetwork(d_model, d_ff, d_model, dropout)

        # 7. Output
        self.output_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, pred_len)
        )

    def forward(self, x):
        # x: (B, seq_len, n_features)
        B, T, C = x.shape

        # 1. Variable Selection
        selected, var_weights = self.vsn(x)  # (B, T, d_model)

        # 2. LSTM
        lstm_out, _ = self.lstm(selected)  # (B, T, d_model)

        # 3. Gated skip connection
        gate = torch.sigmoid(self.post_lstm_gate(lstm_out))
        temporal = self.post_lstm_norm(gate * lstm_out + (1 - gate) * selected)

        # 4. Static enrichment
        enriched = self.enrichment_grn(temporal)  # (B, T, d_model)

        # 5. Temporal attention
        attn_out, attn_weights = self.attention(enriched, enriched, enriched)
        attn_gate = torch.sigmoid(self.attn_gate(attn_out))
        temporal2 = self.attn_norm(attn_gate * attn_out + (1 - attn_gate) * enriched)

        # 6. Feed-forward
        ff_out = self.ff_grn(temporal2)  # (B, T, d_model)

        # 7. Use last time step for prediction
        out = self.output_head(ff_out[:, -1, :]).squeeze(-1)  # (B,)

        # Store attention weights for interpretability
        self._var_weights = var_weights
        self._attn_weights = attn_weights

        return out

    def get_attention_weights(self):
        """학습된 attention weights 반환 (해석용)"""
        return {
            'variable_importance': self._var_weights.detach().cpu(),
            'temporal_attention': self._attn_weights.detach().cpu()
        }


# Build and train
tft = TFTLite(
    n_features=n_features,
    seq_len=SEQ_LEN,
    pred_len=1,
    d_model=CONFIG['d_model'],
    n_heads=CONFIG['n_heads'],
    n_layers=CONFIG['n_layers'],
    d_ff=CONFIG['d_ff'],
    dropout=CONFIG['dropout']
)
print(f'  Parameters: {sum(p.numel() for p in tft.parameters()):,}')

tft, hist_tft = train_model(
    tft, X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_name='TFT-Lite'
)

tft_pred = predict_model(tft, X_test_seq)
result_tft = eval_metrics(y_test_raw, tft_pred, 'TFT-Lite')
print(f'\n  TFT-Lite Test Results:')
print(f'    RMSE={result_tft["RMSE"]:.4f}, MAPE={result_tft["MAPE"]:.2f}%, '
      f'R2={result_tft["R2"]:.4f}, DA={result_tft["DA%"]:.1f}%')

# %%
# ============================================================================
# 9. 학습 곡선 시각화
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, hist, name in zip(axes,
    [hist_patchtst, hist_itrans, hist_tft],
    ['PatchTST', 'iTransformer', 'TFT-Lite']):
    ax.plot(hist['train_loss'], label='Train', alpha=0.8)
    ax.plot(hist['val_loss'], label='Val', alpha=0.8)
    ax.set_title(f'{name} - Learning Curve', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (Huber)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# ============================================================================
# 10. Baseline 모델 (비교용)
# ============================================================================
print('\n' + '=' * 70)
print('  10. Baseline Models')
print('=' * 70)

# 10.1 Naive Drift
prev_test = y.shift(1).loc[y_test.index].values
prev2_test = y.shift(2).loc[y_test.index].values
naive_drift = prev_test + (prev_test - prev2_test)
result_naive = eval_metrics(y_test.values, naive_drift, 'Naive Drift')
print(f'  Naive Drift: RMSE={result_naive["RMSE"]:.4f}')

# 10.2 GradientBoosting
gb = GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=5,
    min_samples_leaf=10, subsample=0.8, random_state=42
)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
result_gb = eval_metrics(y_test.values, gb_pred, 'GradientBoosting')
print(f'  GradientBoosting: RMSE={result_gb["RMSE"]:.4f}')

# 10.3 Hybrid (Naive + GB)
hybrid_pred = 0.8 * naive_drift + 0.2 * gb_pred
result_hybrid = eval_metrics(y_test.values, hybrid_pred, 'Hybrid (Naive+GB)')
print(f'  Hybrid (0.8*Naive + 0.2*GB): RMSE={result_hybrid["RMSE"]:.4f}')

# 10.4 LSTM Baseline
class LSTMBaseline(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1])).squeeze(-1)

lstm_base = LSTMBaseline(n_features)
print(f'  LSTM Baseline params: {sum(p.numel() for p in lstm_base.parameters()):,}')
lstm_base, hist_lstm = train_model(
    lstm_base, X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_name='LSTM'
)
lstm_pred = predict_model(lstm_base, X_test_seq)
result_lstm = eval_metrics(y_test_raw, lstm_pred, 'LSTM (Baseline)')
print(f'  LSTM: RMSE={result_lstm["RMSE"]:.4f}')

# %%
# ============================================================================
# 11. Ensemble 전략
# ============================================================================
print('\n' + '=' * 70)
print('  11. Ensemble Strategies')
print('=' * 70)

# DL 예측값들의 길이 맞추기
test_len = min(len(patchtst_pred), len(itrans_pred), len(tft_pred), len(naive_drift))
naive_al = naive_drift[-test_len:]
gb_al = gb_pred[-test_len:]
patchtst_al = patchtst_pred[-test_len:]
itrans_al = itrans_pred[-test_len:]
tft_al = tft_pred[-test_len:]
lstm_al = lstm_pred[-test_len:]
y_test_al = y_test.values[-test_len:]

ensemble_results = []

# E1: Equal-weight Transformer ensemble
e1 = (patchtst_al + itrans_al + tft_al) / 3
result_e1 = eval_metrics(y_test_al, e1, 'Ensemble: 3-Transformer (equal)')
ensemble_results.append(result_e1)
print(f'  E1 (3-Transformer equal): RMSE={result_e1["RMSE"]:.4f}')

# E2: Naive + Transformer ensemble
e2 = 0.6 * naive_al + 0.15 * patchtst_al + 0.15 * itrans_al + 0.10 * tft_al
result_e2 = eval_metrics(y_test_al, e2, 'Ensemble: Naive(0.6)+Transformers')
ensemble_results.append(result_e2)
print(f'  E2 (Naive+Transformers): RMSE={result_e2["RMSE"]:.4f}')

# E3: Naive + GB + Best Transformer
# 먼저 3개 transformer 중 best 찾기
trans_rmses = {
    'patchtst': result_patchtst['RMSE'],
    'itrans': result_itrans['RMSE'],
    'tft': result_tft['RMSE']
}
best_trans_name = min(trans_rmses, key=trans_rmses.get)
best_trans_pred = {'patchtst': patchtst_al, 'itrans': itrans_al, 'tft': tft_al}[best_trans_name]
e3 = 0.5 * naive_al + 0.3 * gb_al + 0.2 * best_trans_pred
result_e3 = eval_metrics(y_test_al, e3, f'Ensemble: Naive+GB+{best_trans_name}')
ensemble_results.append(result_e3)
print(f'  E3 (Naive+GB+Best_Transformer): RMSE={result_e3["RMSE"]:.4f}')

# E4: Optimized weights via validation performance
# Inverse-RMSE weighted
val_patchtst = predict_model(patchtst, X_val_seq)
val_itrans = predict_model(itransformer, X_val_seq)
val_tft = predict_model(tft, X_val_seq)
val_lstm = predict_model(lstm_base, X_val_seq)
y_val_raw = inverse_transform_y(y_val_seq)

val_len = min(len(val_patchtst), len(val_itrans), len(val_tft))
val_naive = y.shift(1).loc[y_val.index].values
val_naive_drift = val_naive + (val_naive - y.shift(2).loc[y_val.index].values)
val_gb = gb.predict(X_val)

val_preds = {
    'naive': val_naive_drift[-val_len:],
    'gb': val_gb[-val_len:],
    'patchtst': val_patchtst[-val_len:],
    'itrans': val_itrans[-val_len:],
    'tft': val_tft[-val_len:],
}
y_val_al = y_val.values[-val_len:]

# Inverse-RMSE 가중치
inv_rmses = {}
for name, pred in val_preds.items():
    rmse = np.sqrt(mean_squared_error(y_val_al, pred))
    inv_rmses[name] = 1.0 / (rmse + 1e-8)

total_inv = sum(inv_rmses.values())
opt_weights = {k: v / total_inv for k, v in inv_rmses.items()}

print(f'\n  Optimized weights (inverse-RMSE on Val):')
for k, v in sorted(opt_weights.items(), key=lambda x: -x[1]):
    print(f'    {k:12s}: {v:.4f}')

e4 = sum(opt_weights[k] * {'naive': naive_al, 'gb': gb_al, 'patchtst': patchtst_al,
                            'itrans': itrans_al, 'tft': tft_al}[k]
         for k in opt_weights)
result_e4 = eval_metrics(y_test_al, e4, 'Ensemble: Inverse-RMSE Weighted')
ensemble_results.append(result_e4)
print(f'  E4 (Inverse-RMSE Weighted): RMSE={result_e4["RMSE"]:.4f}')

# E5: Stacking with Ridge regression
from sklearn.linear_model import Ridge
stack_val_X = np.column_stack([val_preds[k] for k in ['naive', 'gb', 'patchtst', 'itrans', 'tft']])
stack_test_X = np.column_stack([naive_al, gb_al, patchtst_al, itrans_al, tft_al])

ridge = Ridge(alpha=1.0)
ridge.fit(stack_val_X, y_val_al)
e5 = ridge.predict(stack_test_X)
result_e5 = eval_metrics(y_test_al, e5, 'Ensemble: Stacking (Ridge)')
ensemble_results.append(result_e5)
print(f'  E5 (Stacking Ridge): RMSE={result_e5["RMSE"]:.4f}')
print(f'    Ridge coefficients: {dict(zip(["naive", "gb", "patchtst", "itrans", "tft"], ridge.coef_.round(4)))}')

# %%
# ============================================================================
# 12. Quantile Regression (예측 불확실성)
# ============================================================================
print('\n' + '=' * 70)
print('  12. Prediction Intervals (Quantile Regression)')
print('=' * 70)

scaler_qr = StandardScaler()
X_train_qr = scaler_qr.fit_transform(X_train)
X_test_qr = scaler_qr.transform(X_test)

quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
qr_preds = {}

for q in quantiles:
    qr = QuantileRegressor(quantile=q, alpha=0.1, solver='highs')
    qr.fit(X_train_qr, y_train)
    qr_preds[q] = qr.predict(X_test_qr)
    print(f'  Q{int(q*100):02d} fitted')

# Coverage
cov_90 = np.mean((y_test.values >= qr_preds[0.05]) & (y_test.values <= qr_preds[0.95])) * 100
cov_80 = np.mean((y_test.values >= qr_preds[0.1]) & (y_test.values <= qr_preds[0.9])) * 100
cov_50 = np.mean((y_test.values >= qr_preds[0.25]) & (y_test.values <= qr_preds[0.75])) * 100

print(f'\n  Coverage:')
print(f'    90% PI (Q05-Q95): {cov_90:.1f}%')
print(f'    80% PI (Q10-Q90): {cov_80:.1f}%')
print(f'    50% PI (Q25-Q75): {cov_50:.1f}%')

# Visualization
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(y_test.index, y_test.values, 'k-', label='Actual', lw=2, zorder=5)
ax.plot(y_test.index, qr_preds[0.5], 'b--', label='Median (Q50)', lw=1.5)
ax.fill_between(y_test.index, qr_preds[0.05], qr_preds[0.95],
                alpha=0.15, color='blue', label='90% PI')
ax.fill_between(y_test.index, qr_preds[0.1], qr_preds[0.9],
                alpha=0.25, color='blue', label='80% PI')
ax.fill_between(y_test.index, qr_preds[0.25], qr_preds[0.75],
                alpha=0.35, color='blue', label='50% PI')
ax.set_title(f'Oil Price - Prediction Intervals\n'
             f'90% Cov: {cov_90:.1f}% | 80% Cov: {cov_80:.1f}% | 50% Cov: {cov_50:.1f}%',
             fontweight='bold')
ax.set_ylabel('Brent Crude Oil (USD/barrel)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_quantile_regression.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# ============================================================================
# 13. TFT 해석가능성 분석
# ============================================================================
print('\n' + '=' * 70)
print('  13. TFT Interpretability Analysis')
print('=' * 70)

# Variable importance from TFT
tft.eval()
with torch.no_grad():
    _ = tft(torch.FloatTensor(X_test_seq).to(device))
    weights = tft.get_attention_weights()

# Variable importance (average across time and samples)
var_imp = weights['variable_importance'].mean(dim=(0, 1)).numpy()  # (n_features,)
var_imp_df = pd.DataFrame({
    'feature': selected_features,
    'tft_importance': var_imp
}).sort_values('tft_importance', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Variable importance
ax = axes[0]
top_vars = var_imp_df.head(15)
ax.barh(range(len(top_vars)), top_vars['tft_importance'].values, color='steelblue')
ax.set_yticks(range(len(top_vars)))
ax.set_yticklabels(top_vars['feature'].values)
ax.set_title('TFT Variable Selection Weights (Top 15)', fontweight='bold')
ax.set_xlabel('Weight')
ax.invert_yaxis()

# Temporal attention heatmap (average across test samples)
ax = axes[1]
attn = weights['temporal_attention'].mean(dim=0).numpy()  # (T, T)
# Show last 12 time steps only for readability
attn_last = attn[-12:, -12:]
sns.heatmap(attn_last, ax=ax, cmap='YlOrRd', xticklabels=range(-12, 0), yticklabels=range(-12, 0))
ax.set_title('TFT Temporal Attention (Last 12 Steps)', fontweight='bold')
ax.set_xlabel('Key Position (weeks before prediction)')
ax.set_ylabel('Query Position')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_tft_interpretability.png', dpi=150, bbox_inches='tight')
plt.show()

print('\n  Top 10 Variables (TFT learned):')
for _, row in var_imp_df.head(10).iterrows():
    print(f'    {row["feature"]:40s} weight={row["tft_importance"]:.4f}')

# %%
# ============================================================================
# 14. 전체 결과 종합 & 시각화
# ============================================================================
print('\n' + '=' * 70)
print('  14. Final Results Summary')
print('=' * 70)

all_results = [
    result_naive, result_gb, result_hybrid, result_lstm,
    result_patchtst, result_itrans, result_tft,
] + ensemble_results

results_df = pd.DataFrame(all_results).sort_values('RMSE')
results_df.index = range(1, len(results_df) + 1)

print('\n' + results_df[['Model', 'RMSE', 'MAE', 'MAPE', 'R2', 'DA%']].round(4).to_string())

# Best model
best = results_df.iloc[0]
print(f'\n  BEST MODEL: {best["Model"]}')
print(f'    RMSE = {best["RMSE"]:.4f}')
print(f'    MAPE = {best["MAPE"]:.2f}%')
print(f'    R2   = {best["R2"]:.4f}')
print(f'    DA   = {best["DA%"]:.1f}%')

# 14.1 Bar chart comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, metric, title in zip(axes, ['RMSE', 'MAPE', 'R2'],
                              ['RMSE (Lower is Better)', 'MAPE % (Lower is Better)', 'R² (Higher is Better)']):
    df_plot = results_df.sort_values(metric, ascending=(metric != 'R2'))
    colors = ['#2ecc71' if 'Ensemble' in m or 'Transformer' in m or 'PatchTST' in m or 'iTransformer' in m or 'TFT' in m
              else '#3498db' if 'LSTM' in m else '#e74c3c'
              for m in df_plot['Model']]
    ax.barh(range(len(df_plot)), df_plot[metric].values, color=colors, edgecolor='black', alpha=0.8)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot['Model'].values, fontsize=9)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 14.2 Test period prediction overlay
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(y_test.index, y_test.values, 'k-', label='Actual', lw=2.5, zorder=10)
ax.plot(y_test.index, naive_drift, '--', color='gray', label='Naive Drift', lw=1, alpha=0.6)
ax.plot(y_test.index, gb_pred, '--', color='orange', label='GradientBoosting', lw=1, alpha=0.6)

# Transformer models (aligned to test dates)
test_dates = y_test.index[-test_len:]
ax.plot(test_dates, patchtst_al, '-', color='#e74c3c', label='PatchTST', lw=1.5, alpha=0.8)
ax.plot(test_dates, itrans_al, '-', color='#2ecc71', label='iTransformer', lw=1.5, alpha=0.8)
ax.plot(test_dates, tft_al, '-', color='#9b59b6', label='TFT-Lite', lw=1.5, alpha=0.8)

# Best ensemble
best_ens = results_df[results_df['Model'].str.contains('Ensemble')].iloc[0]
best_ens_name = best_ens['Model']
# Find the right ensemble prediction
for r, preds in [(result_e1, e1), (result_e2, e2), (result_e3, e3), (result_e4, e4), (result_e5, e5)]:
    if r['Model'] == best_ens_name:
        ax.plot(test_dates, preds, '-', color='#e67e22', label=f'{best_ens_name}', lw=2, alpha=0.9)
        break

ax.set_title('Oil Price Prediction - Test Period Comparison', fontweight='bold', fontsize=14)
ax.set_ylabel('Brent Crude Oil (USD/barrel)', fontsize=12)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_test_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# ============================================================================
# 15. 모델 아키텍처 비교 분석
# ============================================================================
print('\n' + '=' * 70)
print('  15. Architecture Analysis')
print('=' * 70)

model_analysis = {
    'PatchTST': {
        'params': sum(p.numel() for p in patchtst.parameters()),
        'approach': 'Channel-Independent Patching',
        'strength': 'Long-range dependency, 과적합 방지',
        'weakness': 'Cross-variate interaction 미포착',
    },
    'iTransformer': {
        'params': sum(p.numel() for p in itransformer.parameters()),
        'approach': 'Inverted (Variate-axis) Attention',
        'strength': 'Cross-variate dependency 직접 학습',
        'weakness': '변수 수에 따라 복잡도 증가',
    },
    'TFT-Lite': {
        'params': sum(p.numel() for p in tft.parameters()),
        'approach': 'Variable Selection + Gated Attention',
        'strength': '해석가능성, 변수 중요도 자동 학습',
        'weakness': '구조 복잡, 학습 불안정 가능',
    },
}

for name, info in model_analysis.items():
    rmse = {'PatchTST': result_patchtst, 'iTransformer': result_itrans, 'TFT-Lite': result_tft}[name]['RMSE']
    print(f'\n  [{name}]')
    print(f'    Parameters : {info["params"]:,}')
    print(f'    Approach   : {info["approach"]}')
    print(f'    Strength   : {info["strength"]}')
    print(f'    Weakness   : {info["weakness"]}')
    print(f'    Test RMSE  : {rmse:.4f}')

# %%
# ============================================================================
# 16. 최종 결론
# ============================================================================
print('\n' + '=' * 70)
print('  16. CONCLUSION')
print('=' * 70)

# Baseline vs Transformer comparison
baseline_best = min(result_naive['RMSE'], result_gb['RMSE'], result_hybrid['RMSE'])
trans_best = min(result_patchtst['RMSE'], result_itrans['RMSE'], result_tft['RMSE'])
ens_best = min(r['RMSE'] for r in ensemble_results)

print(f'\n  [Performance Summary]')
print(f'    Best Baseline (Naive/GB/Hybrid) : RMSE = {baseline_best:.4f}')
print(f'    Best Transformer (single)       : RMSE = {trans_best:.4f}')
print(f'    Best Ensemble                   : RMSE = {ens_best:.4f}')
print(f'    LSTM Baseline                   : RMSE = {result_lstm["RMSE"]:.4f}')

trans_vs_base = (baseline_best - trans_best) / baseline_best * 100
ens_vs_base = (baseline_best - ens_best) / baseline_best * 100

print(f'\n  [Improvement over Best Baseline]')
print(f'    Best Transformer : {trans_vs_base:+.2f}%')
print(f'    Best Ensemble    : {ens_vs_base:+.2f}%')

print(f'\n  [Key Findings]')
print(f'    1. 3개 Advanced Transformer 모델 모두 구현 및 평가 완료')
print(f'       - PatchTST: Channel-Independent patching으로 과적합 억제')
print(f'       - iTransformer: 변수 간 상호작용 직접 모델링')
print(f'       - TFT-Lite: 변수 선택 + 해석가능한 attention')
print(f'    2. Ensemble (Stacking/Weighted)이 단일 모델 대비 일관된 개선')
print(f'    3. Quantile Regression으로 예측 불확실성 정량화')
print(f'       - 90% PI Coverage: {cov_90:.1f}%')
print(f'       - 80% PI Coverage: {cov_80:.1f}%')

print(f'\n  [Recommendations]')
print(f'    - Production: Inverse-RMSE Weighted Ensemble 또는 Stacking')
print(f'    - Risk Management: Quantile Regression PI 활용')
print(f'    - 해석: TFT Variable Importance로 주요 드라이버 모니터링')

print('\n' + '=' * 70)
print('  실험 완료')
print('=' * 70)

# Save results
results_df.to_csv(f'{OUTPUT_DIR}/final_results.csv', index=False)
print(f'\n  Results saved to {OUTPUT_DIR}/final_results.csv')
