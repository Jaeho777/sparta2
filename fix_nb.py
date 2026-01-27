import nbformat as nbf

def fix_notebook(filepath):
    nb = nbf.read(filepath, as_version=4)
    
    # 1. Update Preprocessing Cell to use lagged features (NO LEAKAGE)
    for cell in nb.cells:
        if cell.cell_type == 'code' and '# 1. Data Filtering & Preprocessing' in cell.source:
            cell.source = '''# 1. Data Filtering & Preprocessing
filename = "data_weekly_260120.csv"
df_raw = pd.read_csv(filename)
df_raw["dt"] = pd.to_datetime(df_raw["dt"])
df_raw = df_raw.set_index("dt").sort_index()

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

# Log Returns
df_ret = np.log(df / df.shift(1))

# --- NO LEAKAGE: Lag features by 1 week ---
target_col = "Com_LME_Ni_Cash"
y = df_ret[target_col]
X = df_ret.drop(columns=[target_col]).shift(1) # Lag features

# Align y and X
valid_idx = X.dropna().index.intersection(y.dropna().index)
X = X.loc[valid_idx]
y = y.loc[valid_idx]

print(f"Feature Count: {X.shape[1]}")
print(f"Total Samples (aligned): {len(y)}")
print("Data leakage prevented: Using X(t-1) to predict y(t)")'''

    # 2. Fix the 12-week forecast cell (SYNTAX ERROR FIX)
    for cell in nb.cells:
        if cell.cell_type == 'code' and '5. 12-Week Forecast' in cell.source:
            cell.source = '''# 5. 12-Week Forecast (2025-10-27 to 2026-01-12)
# We look at the actual values and predictions for the last 12 weeks of the dataset
last_12_dates = y_test.index[-12:]
actual_returns = y_test.iloc[-12:]
predicted_returns = y_test_final[-12:]

# Convert log returns back to price relative to start of window
initial_price = df_raw["Com_LME_Ni_Cash"].loc[last_12_dates[0]]
actual_prices = initial_price * np.exp(np.cumsum(actual_returns))
predicted_prices = initial_price * np.exp(np.cumsum(predicted_returns))

mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
print(f"Monthly (approx) MAPE for last 12 weeks: {mape:.2f}%")
print("\\nWeekly Forecast Schedule:")
for d, p in zip(last_12_dates, predicted_prices):
    print(f"{d.date()}: {p:.2f}")'''

    nbf.write(nb, filepath)
    print(f"Successfully fixed {filepath}")

if __name__ == "__main__":
    fix_notebook('sparta2.ipynb')
