import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import os
import matplotlib.pyplot as plt

# ============================================================
#  CONFIG
# ============================================================
CONFIG = {
    'data_file': 'data_weekly_260120.csv',
    'target_col': 'Com_LME_Ni_Cash',
    'val_start': '2025-08-04',
    'random_seed': 42,
    'exclude_lme_index': True
}

# Seed
np.random.seed(CONFIG['random_seed'])

print("Loading data...")
# 1. Load and Preprocess Data
try:
    df_raw = pd.read_csv(CONFIG['data_file'])
    df_raw["dt"] = pd.to_datetime(df_raw["dt"])
    df_raw = df_raw.set_index("dt").sort_index()
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

def filter_cols(columns):
    target = "Com_LME_Ni_Cash"
    metals = ["Gold", "Silver", "Iron", "Steel", "Copper", "Aluminum", "Zinc", "Nickel", "Lead", "Tin", "Uranium"]
    filtered = [target]
    for col in columns:
        if any(x in col for x in ["Idx_", "Bonds_", "EX_"]):
            filtered.append(col)
        elif "Com_LME" in col:
            filtered.append(col)
        elif any(m in col for m in metals):
            filtered.append(col)
    return sorted(list(set(filtered)))

filtered_cols = filter_cols(df_raw.columns)
df = df_raw[filtered_cols].copy()
df = df.ffill().bfill()

# Lag features to prevent leakage (Match notebook logic)
target_col = "Com_LME_Ni_Cash"
y = df[target_col]
X = df.drop(columns=[target_col]).shift(1)

# Align
valid_idx = X.dropna().index.intersection(y.dropna().index)
X = X.loc[valid_idx]
y = y.loc[valid_idx]

print(f"Using Raw Features: {X.shape[1]}")

# Split Train (Match notebook logic)
train_mask = X.index < CONFIG['val_start']
X_train_all = X[train_mask]
y_train = y[train_mask]

print(f"Train samples: {len(X_train_all)}")

# ============================================================
# SHAP Feature Selection (Random Probe Method)
# ============================================================
print("\nRunning SHAP Feature Selection with Random Probe Method...")

# Add Random Probe
X_train_probe = X_train_all.copy()
X_train_probe['random_probe'] = np.random.normal(0, 1, size=len(X_train_probe))

model_shap = xgb.XGBRegressor(
    n_estimators=100, 
    random_state=CONFIG['random_seed'], 
    n_jobs=-1
)
model_shap.fit(X_train_probe, y_train)

explainer = shap.TreeExplainer(model_shap)
shap_val = explainer.shap_values(X_train_probe)
importances = np.abs(shap_val).mean(axis=0)

feat_imp = pd.DataFrame({
    "feature": X_train_probe.columns, 
    "importance": importances
}).sort_values("importance", ascending=False)

# Threshold 결정
probe_importance = feat_imp[feat_imp['feature'] == 'random_probe']['importance'].values[0]
print(f"Random Probe Importance (Threshold): {probe_importance:.6f}")

# Select features better than noise
selected_features = feat_imp[feat_imp['importance'] > probe_importance]['feature'].tolist()
selected_features = [f for f in selected_features if f != 'random_probe']

# Remove LME Index if requested (Notebook logic)
if CONFIG['exclude_lme_index']:
    original_len = len(selected_features)
    selected_features = [f for f in selected_features if 'Com_LME_Index' not in f]
    if len(selected_features) < original_len:
        print("Excluded Com_LME_Index related features.")

# Safety Net
if len(selected_features) < 5:
    print("Warning: Too few features selected. selecting top 5 from regular ranking.")
    # Fallback to top 5 non-probe features
    selected_features = feat_imp[feat_imp['feature'] != 'random_probe'].head(5)['feature'].tolist()

print(f"\nSelected {len(selected_features)} features:")
for i, f in enumerate(selected_features):
    imp = feat_imp[feat_imp['feature'] == f]['importance'].values[0]
    print(f"  {i+1}. {f} (SHAP: {imp:.4f})")

# Save to file
with open('selected_features.txt', 'w') as f:
    for item in selected_features:
        f.write(f"{item}\n")

print("\nSaved selected features to selected_features.txt")

# Generate Image
try:
    print("Generating SHAP summary image...")
    plt.figure(figsize=(10, 6))
    # Top 10 for visualization (excluding probe if it were there, but selected_features is clean)
    # We need importance values for selected features
    top_features = selected_features[:10]
    top_importances = []
    for f in top_features:
        top_importances.append(feat_imp[feat_imp['feature'] == f]['importance'].values[0])
    
    # Sort for barh (ascending)
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_importances[::-1], align='center', color='skyblue')
    plt.yticks(y_pos, top_features[::-1])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Top 10 Features (Random Probe Selection)')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150)
    print("Saved shap_summary.png")
except Exception as e:
    print(f"Error generating image: {e}")
