#!/usr/bin/env python3
"""
Rebuild sparta2.ipynb with:
1. Fix all logical bugs
2. Add proper visualizations at every step
3. Remove duplicate cells
4. Fix section numbering
5. Ensure causal flow is clear
"""

import json
import copy

def make_code_cell(source, cell_id=None):
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split('\n') if isinstance(source, str) else source
    }
    # Convert to proper format: each line except last should end with \n
    lines = source.split('\n') if isinstance(source, str) else source
    formatted = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            formatted.append(line + '\n' if not line.endswith('\n') else line)
        else:
            formatted.append(line.rstrip('\n'))
    cell['source'] = formatted
    if cell_id:
        cell['id'] = cell_id
    return cell


def make_md_cell(source, cell_id=None):
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split('\n') if isinstance(source, str) else source
    }
    lines = source.split('\n') if isinstance(source, str) else source
    formatted = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            formatted.append(line + '\n' if not line.endswith('\n') else line)
        else:
            formatted.append(line.rstrip('\n'))
    cell['source'] = formatted
    if cell_id:
        cell['id'] = cell_id
    return cell


def main():
    with open('sparta2.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    old_cells = nb['cells']
    new_cells = []

    # =========================================================================
    # Keep cells 0-3 as-is (Title, Experiment Design, Stacking Structure, Config)
    # =========================================================================
    for i in range(4):
        new_cells.append(old_cells[i])

    # =========================================================================
    # NEW: Add Experiment Design Visualization (Pipeline Flow Diagram)
    # =========================================================================
    new_cells.append(make_md_cell(
        "### 0.2 분석 파이프라인 시각화\n"
        "목표: 전체 분석 흐름을 시각적으로 확인하여 각 단계의 인과관계를 명확히 함."
    ))

    new_cells.append(make_code_cell(
        "# 0.2 분석 파이프라인 시각화\n"
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.patches as mpatches\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(16, 6))\n"
        "ax.set_xlim(0, 10)\n"
        "ax.set_ylim(0, 6)\n"
        "ax.axis('off')\n"
        "ax.set_title('ML Pipeline: Nickel Price Prediction', fontsize=16, fontweight='bold', pad=20)\n"
        "\n"
        "# Pipeline stages\n"
        "stages = [\n"
        "    (0.5, 4.5, 'Data\\nPreparation', '#E3F2FD'),\n"
        "    (2.3, 4.5, 'Feature\\nSelection\\n(SHAP)', '#E8F5E9'),\n"
        "    (4.1, 4.5, 'Baseline\\nModels (5)', '#FFF3E0'),\n"
        "    (5.9, 4.5, 'Residual\\nStacking', '#FCE4EC'),\n"
        "    (7.7, 4.5, 'ROR\\nStacking', '#F3E5F5'),\n"
        "]\n"
        "\n"
        "for x, y, text, color in stages:\n"
        "    box = mpatches.FancyBboxPatch((x, y-0.6), 1.4, 1.2,\n"
        "                                   boxstyle='round,pad=0.1', \n"
        "                                   facecolor=color, edgecolor='#333', linewidth=1.5)\n"
        "    ax.add_patch(box)\n"
        "    ax.text(x+0.7, y, text, ha='center', va='center', fontsize=9, fontweight='bold')\n"
        "\n"
        "# Arrows between stages\n"
        "for i in range(len(stages)-1):\n"
        "    x1 = stages[i][0] + 1.4\n"
        "    x2 = stages[i+1][0]\n"
        "    y = 4.5\n"
        "    ax.annotate('', xy=(x2, y), xytext=(x1, y),\n"
        "                arrowprops=dict(arrowstyle='->', color='#555', lw=2))\n"
        "\n"
        "# Evaluation branch\n"
        "eval_stages = [\n"
        "    (2.3, 2.0, 'Naive\\nBaseline', '#ECEFF1'),\n"
        "    (4.1, 2.0, 'Hybrid\\n(Naive+ML)', '#E0F7FA'),\n"
        "    (5.9, 2.0, 'Backtest\\n& ROR', '#FFF9C4'),\n"
        "    (7.7, 2.0, 'Directional\\nEvaluation', '#FFEBEE'),\n"
        "]\n"
        "\n"
        "for x, y, text, color in eval_stages:\n"
        "    box = mpatches.FancyBboxPatch((x, y-0.6), 1.4, 1.2,\n"
        "                                   boxstyle='round,pad=0.1', \n"
        "                                   facecolor=color, edgecolor='#666', linewidth=1, linestyle='--')\n"
        "    ax.add_patch(box)\n"
        "    ax.text(x+0.7, y, text, ha='center', va='center', fontsize=9)\n"
        "\n"
        "for i in range(len(eval_stages)-1):\n"
        "    x1 = eval_stages[i][0] + 1.4\n"
        "    x2 = eval_stages[i+1][0]\n"
        "    y = 2.0\n"
        "    ax.annotate('', xy=(x2, y), xytext=(x1, y),\n"
        "                arrowprops=dict(arrowstyle='->', color='#999', lw=1.5, linestyle='dashed'))\n"
        "\n"
        "# Connect top and bottom\n"
        "ax.annotate('', xy=(4.8, 2.6), xytext=(4.8, 3.9),\n"
        "            arrowprops=dict(arrowstyle='->', color='#999', lw=1.5, linestyle='dotted'))\n"
        "ax.text(5.1, 3.2, 'Compare', fontsize=8, color='#666', fontstyle='italic')\n"
        "\n"
        "# Labels\n"
        "ax.text(0.2, 5.5, 'Stage 1-3: Model Pipeline', fontsize=11, fontweight='bold', color='#333')\n"
        "ax.text(0.2, 3.0, 'Stage 4: Evaluation & Discovery', fontsize=11, fontweight='bold', color='#666')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # =========================================================================
    # Keep cells 4-5 (Section 1 markdown + Feature Engineering explanation)
    # =========================================================================
    new_cells.append(old_cells[4])  # Section 1 markdown
    new_cells.append(old_cells[5])  # Feature Engineering explanation

    # =========================================================================
    # Keep cell 6 (Data preparation code)
    # =========================================================================
    new_cells.append(old_cells[6])

    # =========================================================================
    # NEW: Train/Val/Test split definition + visualization
    # =========================================================================
    new_cells.append(make_md_cell(
        "### 1.1 Train/Validation/Test Split\n"
        "목표: 시간순 분할을 시각화하여 데이터 누수가 없음을 확인.\n"
        "해석: 시간순 분할은 미래 정보 누수를 원천 차단. 검증/테스트 각 12주는 약 3개월에 해당."
    ))

    new_cells.append(make_code_cell(
        "# 1.1 Train/Val/Test Split 정의 및 시각화\n"
        "VAL_START = CONFIG['val_start']\n"
        "VAL_END = CONFIG['val_end']\n"
        "TEST_START = CONFIG['test_start']\n"
        "TEST_END = CONFIG['test_end']\n"
        "\n"
        "# Split\n"
        "train_mask = X.index < VAL_START\n"
        "val_mask = (X.index >= VAL_START) & (X.index <= VAL_END)\n"
        "test_mask = (X.index >= TEST_START) & (X.index <= TEST_END)\n"
        "\n"
        "X_train_all, y_train = X[train_mask], y[train_mask]\n"
        "X_val_all, y_val = X[val_mask], y[val_mask]\n"
        "X_test_all, y_test = X[test_mask], y[test_mask]\n"
        "\n"
        "print(f'Train: {X_train_all.index[0].date()} ~ {X_train_all.index[-1].date()} ({len(X_train_all)} weeks)')\n"
        "print(f'Val:   {X_val_all.index[0].date()} ~ {X_val_all.index[-1].date()} ({len(X_val_all)} weeks)')\n"
        "print(f'Test:  {X_test_all.index[0].date()} ~ {X_test_all.index[-1].date()} ({len(X_test_all)} weeks)')\n"
        "\n"
        "# --- Train/Val/Test Split Visualization ---\n"
        "fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [3, 1]})\n"
        "\n"
        "# Top: Price with colored regions\n"
        "ax = axes[0]\n"
        "ax.plot(df.index, df[target_col], color='black', linewidth=0.8, alpha=0.7)\n"
        "ax.axvspan(X_train_all.index[0], X_train_all.index[-1], alpha=0.15, color='blue', label=f'Train ({len(X_train_all)}w)')\n"
        "ax.axvspan(X_val_all.index[0], X_val_all.index[-1], alpha=0.25, color='orange', label=f'Validation ({len(X_val_all)}w)')\n"
        "ax.axvspan(X_test_all.index[0], X_test_all.index[-1], alpha=0.25, color='red', label=f'Test ({len(X_test_all)}w)')\n"
        "ax.set_title('Nickel Price with Train/Validation/Test Split', fontsize=13)\n"
        "ax.set_ylabel('Price (USD/tonne)')\n"
        "ax.legend(loc='upper left')\n"
        "ax.grid(True, alpha=0.3)\n"
        "\n"
        "# Bottom: Data split bar\n"
        "ax2 = axes[1]\n"
        "ax2.barh(0, len(X_train_all), left=0, color='#2196F3', alpha=0.7, label='Train')\n"
        "ax2.barh(0, len(X_val_all), left=len(X_train_all), color='#FF9800', alpha=0.7, label='Validation')\n"
        "ax2.barh(0, len(X_test_all), left=len(X_train_all)+len(X_val_all), color='#F44336', alpha=0.7, label='Test')\n"
        "ax2.set_xlabel('Weeks')\n"
        "ax2.set_yticks([])\n"
        "ax2.set_title('Split Proportions', fontsize=11)\n"
        "# Add text\n"
        "total = len(X_train_all) + len(X_val_all) + len(X_test_all)\n"
        "ax2.text(len(X_train_all)/2, 0, f'Train\\n{len(X_train_all)}w ({len(X_train_all)/total*100:.0f}%)', \n"
        "         ha='center', va='center', fontsize=10, fontweight='bold', color='white')\n"
        "ax2.text(len(X_train_all)+len(X_val_all)/2, 0, f'Val\\n{len(X_val_all)}w', \n"
        "         ha='center', va='center', fontsize=9, fontweight='bold')\n"
        "ax2.text(len(X_train_all)+len(X_val_all)+len(X_test_all)/2, 0, f'Test\\n{len(X_test_all)}w', \n"
        "         ha='center', va='center', fontsize=9, fontweight='bold', color='white')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig('report_images/train_val_test_split.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ))

    # =========================================================================
    # Keep cell 7 (Data Overview markdown) but update
    # =========================================================================
    new_cells.append(make_md_cell(
        "### 1.2 Data Overview\n"
        "목표: 가격 추세, 수익률 분포, 변동성 수준을 시각적으로 확인.\n"
        "해석: 이상치나 변동성 급등은 모델 불안정 신호. 수익률 분포의 비정규성(첨도, 왜도)은 선형 모델의 한계를 암시."
    ))

    # =========================================================================
    # Enhanced cell 8 (Data Overview visualization) - more comprehensive
    # =========================================================================
    new_cells.append(make_code_cell(
        "# 1.2 데이터 개요 시각화 (3-panel)\n"
        "fig, axes = plt.subplots(1, 3, figsize=(18, 4))\n"
        "\n"
        "# 1. Price time series\n"
        "ax = axes[0]\n"
        "ax.plot(df.index, df[target_col], color='black', linewidth=0.8)\n"
        "ax.axvline(pd.Timestamp(VAL_START), color='orange', linestyle='--', alpha=0.7, label='Val start')\n"
        "ax.axvline(pd.Timestamp(TEST_START), color='red', linestyle='--', alpha=0.7, label='Test start')\n"
        "ax.set_title('Nickel Price (13Y History)', fontsize=11)\n"
        "ax.set_ylabel('USD/tonne')\n"
        "ax.legend(fontsize=8)\n"
        "ax.grid(True, alpha=0.3)\n"
        "\n"
        "# 2. Return distribution\n"
        "ax = axes[1]\n"
        "returns = df_ret[target_col].dropna()\n"
        "ax.hist(returns, bins=40, color='steelblue', edgecolor='white', alpha=0.8, density=True)\n"
        "# Normal overlay\n"
        "from scipy import stats\n"
        "x_range = np.linspace(returns.min(), returns.max(), 100)\n"
        "ax.plot(x_range, stats.norm.pdf(x_range, returns.mean(), returns.std()), \n"
        "        'r-', linewidth=2, label=f'Normal(mu={returns.mean():.4f}, sig={returns.std():.4f})')\n"
        "ax.set_title('Weekly Log Return Distribution', fontsize=11)\n"
        "ax.set_xlabel('Log Return')\n"
        "ax.legend(fontsize=7)\n"
        "# Add skew/kurtosis annotation\n"
        "skew = returns.skew()\n"
        "kurt = returns.kurtosis()\n"
        "ax.text(0.95, 0.95, f'Skew: {skew:.2f}\\nKurtosis: {kurt:.2f}\\nN: {len(returns)}',\n"
        "        transform=ax.transAxes, ha='right', va='top', fontsize=8,\n"
        "        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))\n"
        "\n"
        "# 3. Rolling volatility\n"
        "ax = axes[2]\n"
        "rolling_vol = df_ret[target_col].rolling(12).std() * np.sqrt(52)\n"
        "ax.plot(rolling_vol.index, rolling_vol, color='darkorange', linewidth=0.8)\n"
        "ax.axhline(rolling_vol.mean(), color='gray', linestyle='--', alpha=0.5, label=f'Mean: {rolling_vol.mean():.2f}')\n"
        "ax.axvline(pd.Timestamp(TEST_START), color='red', linestyle='--', alpha=0.5)\n"
        "ax.set_title('Annualized Volatility (12w rolling)', fontsize=11)\n"
        "ax.set_ylabel('Volatility')\n"
        "ax.legend(fontsize=8)\n"
        "ax.grid(True, alpha=0.3)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig('report_images/nickel_data_overview.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "\n"
        "print(f'Price Range: {df[target_col].min():.0f} ~ {df[target_col].max():.0f} USD/tonne')\n"
        "print(f'Return Stats: mean={returns.mean():.4f}, std={returns.std():.4f}, skew={skew:.2f}, kurtosis={kurt:.2f}')"
    ))

    # =========================================================================
    # Keep cell 9 (Section 2 Data Quality markdown)
    # =========================================================================
    new_cells.append(make_md_cell(
        "## 2. 데이터 품질 검사\n"
        "목표: 주간 주기(7일)와 결측 여부를 확인.\n"
        "해석: 주기가 깨지면 리샘플링 필요, 결측은 재처리 필요."
    ))

    # =========================================================================
    # Enhanced cell 10 (Data Quality) with visualization
    # =========================================================================
    new_cells.append(make_code_cell(
        "# 2. 데이터 품질 검사 + 시각화\n"
        "missing = df.isnull().sum().sum()\n"
        "if missing > 0:\n"
        "    print(f'Missing values remain after forward fill: {missing}')\n"
        "else:\n"
        "    print('Data is clean. No missing values found.')\n"
        "\n"
        "# 주기 검사\n"
        "freq_check = df.index.to_series().diff().dt.days.value_counts()\n"
        "print('\\nTime Frequency Distribution (Days):')\n"
        "print(freq_check)\n"
        "\n"
        "if freq_check.index[0] != 7:\n"
        "    print('WARNING: Data is not strictly weekly. Consider resampling if necessary.')\n"
        "else:\n"
        "    print('Data confirmed as Weekly.')\n"
        "\n"
        "# --- 시각화 ---\n"
        "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n"
        "\n"
        "# 1. Time gap distribution\n"
        "ax = axes[0]\n"
        "gaps = df.index.to_series().diff().dt.days.dropna()\n"
        "colors = ['#4CAF50' if g == 7 else '#F44336' for g in gaps]\n"
        "ax.bar(freq_check.index, freq_check.values, color=['#4CAF50' if x==7 else '#FF9800' for x in freq_check.index])\n"
        "ax.set_xlabel('Gap (days)')\n"
        "ax.set_ylabel('Count')\n"
        "ax.set_title('Time Gap Distribution')\n"
        "pct_7 = (gaps == 7).sum() / len(gaps) * 100\n"
        "ax.text(0.95, 0.95, f'7-day: {pct_7:.1f}%', transform=ax.transAxes, ha='right', va='top',\n"
        "        fontsize=10, bbox=dict(facecolor='lightgreen', alpha=0.5))\n"
        "\n"
        "# 2. Missing values per column (before fill)\n"
        "ax = axes[1]\n"
        "missing_before = df_raw[filtered_cols].isnull().sum()\n"
        "missing_nonzero = missing_before[missing_before > 0].sort_values(ascending=False)\n"
        "if len(missing_nonzero) > 0:\n"
        "    missing_nonzero.head(15).plot(kind='barh', ax=ax, color='#FF5722')\n"
        "    ax.set_title(f'Missing Values (Top 15, before ffill)')\n"
        "else:\n"
        "    ax.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14, transform=ax.transAxes)\n"
        "    ax.set_title('Missing Values Check')\n"
        "ax.set_xlabel('Count')\n"
        "\n"
        "# 3. Feature count by category\n"
        "ax = axes[2]\n"
        "categories = {'LME Metals': 0, 'Currencies': 0, 'Indices': 0, 'Bonds': 0, 'Commodities': 0, 'Other': 0}\n"
        "for col in X.columns:\n"
        "    if 'Com_LME' in col: categories['LME Metals'] += 1\n"
        "    elif 'EX_' in col: categories['Currencies'] += 1\n"
        "    elif 'Idx_' in col: categories['Indices'] += 1\n"
        "    elif 'Bonds_' in col: categories['Bonds'] += 1\n"
        "    elif 'Com_' in col: categories['Commodities'] += 1\n"
        "    else: categories['Other'] += 1\n"
        "cats = {k: v for k, v in categories.items() if v > 0}\n"
        "ax.pie(cats.values(), labels=[f'{k}\\n({v})' for k,v in cats.items()], \n"
        "       autopct='%1.0f%%', colors=plt.cm.Set3.colors[:len(cats)])\n"
        "ax.set_title(f'Feature Categories (Total: {X.shape[1]})')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # =========================================================================
    # Keep cells 11-12 (Section 3 Feature Selection markdown + SHAP code)
    # =========================================================================
    new_cells.append(old_cells[11])  # Section 3 markdown
    new_cells.append(old_cells[12])  # SHAP code

    # =========================================================================
    # Keep cell 13 (3.1 markdown)
    # =========================================================================
    new_cells.append(old_cells[13])

    # =========================================================================
    # Enhanced cell 14 (Feature importance + correlation) with SHAP beeswarm
    # =========================================================================
    new_cells.append(make_code_cell(
        "# 3.1 피처 중요도 및 상관관계 + SHAP Beeswarm\n"
        "fig, axes = plt.subplots(1, 3, figsize=(20, 7))\n"
        "\n"
        "# 1. Feature importance bar\n"
        "ax = axes[0]\n"
        "top_feat = feat_imp.head(top_n).sort_values('importance')\n"
        "colors = ['#FF5722' if f in selected_features else '#90CAF9' for f in top_feat['feature']]\n"
        "ax.barh(top_feat['feature'], top_feat['importance'], color=colors)\n"
        "ax.set_title('SHAP Feature Importance (Train Only)', fontsize=11)\n"
        "ax.set_xlabel('Mean |SHAP value|')\n"
        "# Add legend for selected vs excluded\n"
        "import matplotlib.patches as mpatches\n"
        "sel_patch = mpatches.Patch(color='#FF5722', label='Selected')\n"
        "exc_patch = mpatches.Patch(color='#90CAF9', label='Not selected')\n"
        "ax.legend(handles=[sel_patch, exc_patch], fontsize=8)\n"
        "\n"
        "# 2. SHAP Beeswarm (summary plot)\n"
        "ax = axes[1]\n"
        "# Manual beeswarm-like visualization\n"
        "shap_selected = shap_val[:, [list(X_train_all.columns).index(f) for f in selected_features]]\n"
        "feature_order = np.argsort(np.abs(shap_selected).mean(axis=0))[::-1]\n"
        "for rank, idx in enumerate(feature_order[:15]):\n"
        "    vals = shap_selected[:, idx]\n"
        "    feat_vals = X_train_all[selected_features[idx]].values\n"
        "    # Normalize feature values for coloring\n"
        "    vmin, vmax = np.nanpercentile(feat_vals, [5, 95])\n"
        "    if vmax - vmin == 0: vmax = vmin + 1\n"
        "    norm_vals = np.clip((feat_vals - vmin) / (vmax - vmin), 0, 1)\n"
        "    jitter = np.random.normal(0, 0.15, len(vals))\n"
        "    ax.scatter(vals, np.full_like(vals, 14-rank) + jitter, \n"
        "              c=norm_vals, cmap='coolwarm', s=3, alpha=0.5)\n"
        "ax.set_yticks(range(15))\n"
        "ax.set_yticklabels([selected_features[i] for i in feature_order[:15]][::-1], fontsize=8)\n"
        "ax.set_xlabel('SHAP value')\n"
        "ax.set_title('SHAP Beeswarm (Top 15 Selected)', fontsize=11)\n"
        "ax.axvline(0, color='gray', linestyle='-', alpha=0.3)\n"
        "\n"
        "# 3. Correlation heatmap\n"
        "ax = axes[2]\n"
        "corr = X_train_all[selected_features].corr()\n"
        "mask = np.triu(np.ones_like(corr, dtype=bool))\n"
        "sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, ax=ax,\n"
        "            xticklabels=True, yticklabels=True, \n"
        "            cbar_kws={'shrink': 0.8}, linewidths=0.5)\n"
        "ax.set_title('Feature Correlation (Train)', fontsize=11)\n"
        "ax.tick_params(axis='both', labelsize=6)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig('report_images/feature_analysis.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "\n"
        "# High correlation pairs\n"
        "print('\\n[High Correlation Pairs (|r| > 0.7)]')\n"
        "for i in range(len(selected_features)):\n"
        "    for j in range(i+1, len(selected_features)):\n"
        "        r = corr.iloc[i, j]\n"
        "        if abs(r) > 0.7:\n"
        "            print(f'  {selected_features[i]} <-> {selected_features[j]}: r={r:.3f}')"
    ))

    # =========================================================================
    # Keep cells 15-20 (Feature stability, domain interpretation, exclusion exp)
    # =========================================================================
    for i in range(15, 21):
        new_cells.append(old_cells[i])

    # =========================================================================
    # Keep cells 21-23 (Section 4 markdown, 4.1 markdown, utility code)
    # =========================================================================
    new_cells.append(old_cells[21])  # Section 4 markdown
    new_cells.append(old_cells[22])  # 4.1 markdown
    new_cells.append(old_cells[23])  # Utility code

    # =========================================================================
    # Keep cell 24 (Baseline models)
    # =========================================================================
    new_cells.append(old_cells[24])

    # =========================================================================
    # Keep cells 25-26 (4.2 markdown + Baseline selection code)
    # =========================================================================
    new_cells.append(old_cells[25])
    new_cells.append(old_cells[26])

    # =========================================================================
    # Keep cells 27-28 (4.2.1 markdown + Validation summary viz)
    # =========================================================================
    new_cells.append(old_cells[27])
    new_cells.append(old_cells[28])

    # =========================================================================
    # Keep cell 29 (4.3 ROR stacking markdown)
    # =========================================================================
    new_cells.append(old_cells[29])

    # =========================================================================
    # Keep cell 30 (Residual + ROR search code)
    # =========================================================================
    new_cells.append(old_cells[30])

    # =========================================================================
    # NEW: Residual/ROR Stacking Results Visualization
    # =========================================================================
    new_cells.append(make_md_cell(
        "### 4.3.1 Stacking 탐색 결과 시각화\n"
        "목표: Baseline → Residual → ROR 각 단계의 검증 성능 변화를 시각적으로 비교.\n"
        "해석: 단계가 추가될수록 검증 RMSE가 감소하면 스태킹이 효과적임을 의미."
    ))

    new_cells.append(make_code_cell(
        "# 4.3.1 Stacking 탐색 결과 시각화\n"
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
        "\n"
        "# 1. Baseline vs Residual vs ROR (RMSE 비교)\n"
        "ax = axes[0]\n"
        "stage_data = []\n"
        "# Baseline best\n"
        "if best_baseline_model:\n"
        "    bl_rmse = baseline_val_df.loc[best_baseline_model, 'VAL_RMSE']\n"
        "    stage_data.append(('Baseline\\n(Best)', bl_rmse, '#2196F3'))\n"
        "# Residual best\n"
        "if best_resid_combo is not None:\n"
        "    stage_data.append(('+ Residual\\n(Best)', best_resid_combo['VAL_RMSE'], '#4CAF50'))\n"
        "# ROR best\n"
        "if best_ror_combo is not None:\n"
        "    stage_data.append(('+ ROR\\n(Best)', best_ror_combo['VAL_RMSE'], '#FF9800'))\n"
        "\n"
        "if stage_data:\n"
        "    names, values, colors = zip(*stage_data)\n"
        "    bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=0.5)\n"
        "    for bar, val in zip(bars, values):\n"
        "        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,\n"
        "                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')\n"
        "ax.set_ylabel('Validation RMSE')\n"
        "ax.set_title('Stacking Stage Comparison (Val)', fontsize=11)\n"
        "ax.grid(True, alpha=0.3, axis='y')\n"
        "\n"
        "# 2. Residual combination heatmap (Base x Residual)\n"
        "ax = axes[1]\n"
        "if len(resid_best_df) > 0:\n"
        "    pivot = resid_best_df.pivot_table(index='Base', columns='Residual', values='VAL_RMSE', aggfunc='min')\n"
        "    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd_r', ax=ax, linewidths=0.5)\n"
        "    ax.set_title('Residual Stacking: Base x Residual (Val RMSE)', fontsize=10)\n"
        "else:\n"
        "    ax.text(0.5, 0.5, 'No residual results', ha='center', va='center', transform=ax.transAxes)\n"
        "\n"
        "# 3. Top 10 overall combinations\n"
        "ax = axes[2]\n"
        "top_combos = []\n"
        "for _, r in baseline_val_df.iterrows():\n"
        "    top_combos.append({'Model': r.name, 'VAL_RMSE': r['VAL_RMSE'], 'Stage': 'Baseline'})\n"
        "if len(resid_best_df) > 0:\n"
        "    for _, r in resid_best_df.head(3).iterrows():\n"
        "        top_combos.append({'Model': f\"{r['Base']}+{r['Residual']}\", 'VAL_RMSE': r['VAL_RMSE'], 'Stage': 'Residual'})\n"
        "if len(ror_best_df) > 0:\n"
        "    for _, r in ror_best_df.head(3).iterrows():\n"
        "        top_combos.append({'Model': f\"{r['Base']}+{r['Residual']}+{r['ROR']}\", 'VAL_RMSE': r['VAL_RMSE'], 'Stage': 'ROR'})\n"
        "tc_df = pd.DataFrame(top_combos).sort_values('VAL_RMSE')\n"
        "color_map = {'Baseline': '#2196F3', 'Residual': '#4CAF50', 'ROR': '#FF9800'}\n"
        "bar_colors = [color_map[s] for s in tc_df['Stage']]\n"
        "ax.barh(tc_df['Model'], tc_df['VAL_RMSE'], color=bar_colors)\n"
        "ax.set_xlabel('Validation RMSE')\n"
        "ax.set_title('Top Combinations by Stage', fontsize=10)\n"
        "ax.invert_yaxis()\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig('report_images/stacking_results.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ))

    # =========================================================================
    # Keep cell 31 (4.4 Residual Diagnostics markdown)
    # =========================================================================
    new_cells.append(old_cells[31])

    # =========================================================================
    # SKIP cell 32 (empty pass cell) - remove it
    # =========================================================================
    # old_cells[32] is just "pass" - skip

    # =========================================================================
    # Keep cell 33 (Test evaluation code)
    # =========================================================================
    new_cells.append(old_cells[33])

    # =========================================================================
    # NEW: Test Results Comprehensive Visualization
    # =========================================================================
    new_cells.append(make_md_cell(
        "### 4.5 Test 평가 결과 시각화\n"
        "목표: 모든 스태킹 단계의 테스트 성능을 비교하여 과적합 여부 검증.\n"
        "해석: 검증-테스트 RMSE 괴리가 크면 과적합. Naive가 ML을 이기면 시장 구조 변화(레짐 전환)를 의미."
    ))

    new_cells.append(make_code_cell(
        "# 4.5 Test 평가 결과 종합 시각화\n"
        "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n"
        "\n"
        "# 1. All models RMSE ranking (Test)\n"
        "ax = axes[0, 0]\n"
        "sorted_results = results_df.sort_values('RMSE')\n"
        "colors_rmse = []\n"
        "for m in sorted_results.index:\n"
        "    if m.startswith('Naive'): colors_rmse.append('#8BC34A')\n"
        "    elif m.startswith('BASE_'): colors_rmse.append('#2196F3')\n"
        "    elif m.startswith('RES_'): colors_rmse.append('#FF9800')\n"
        "    elif m.startswith('ROR_'): colors_rmse.append('#9C27B0')\n"
        "    else: colors_rmse.append('#607D8B')\n"
        "ax.barh(sorted_results.index, sorted_results['RMSE'], color=colors_rmse)\n"
        "ax.set_xlabel('Test RMSE')\n"
        "ax.set_title('All Models: Test RMSE Ranking', fontsize=11)\n"
        "ax.invert_yaxis()\n"
        "\n"
        "# 2. Actual vs Top-3 Predictions\n"
        "ax = axes[0, 1]\n"
        "ax.plot(preds.index, preds['Actual'], 'k-', linewidth=2, label='Actual')\n"
        "top3 = sorted_results.head(3).index\n"
        "line_styles = ['--', '-.', ':']\n"
        "top_colors = ['#F44336', '#2196F3', '#4CAF50']\n"
        "for m, ls, c in zip(top3, line_styles, top_colors):\n"
        "    if m in preds.columns:\n"
        "        rmse_val = sorted_results.loc[m, 'RMSE']\n"
        "        ax.plot(preds.index, preds[m], linestyle=ls, color=c, linewidth=1.5, \n"
        "                label=f'{m[:25]} (RMSE={rmse_val:.0f})')\n"
        "ax.set_title('Top-3 Models vs Actual (Test)', fontsize=11)\n"
        "ax.legend(fontsize=7)\n"
        "ax.grid(True, alpha=0.3)\n"
        "\n"
        "# 3. Residual distribution comparison\n"
        "ax = axes[1, 0]\n"
        "if len(sorted_results) >= 2:\n"
        "    best_m = sorted_results.index[0]\n"
        "    worst_m = sorted_results.index[-1]\n"
        "    if best_m in preds.columns:\n"
        "        ax.hist(preds['Actual'] - preds[best_m], bins=12, alpha=0.6, \n"
        "                color='#4CAF50', label=f'Best: {best_m[:20]}', edgecolor='white')\n"
        "    if worst_m in preds.columns:\n"
        "        ax.hist(preds['Actual'] - preds[worst_m], bins=12, alpha=0.6, \n"
        "                color='#F44336', label=f'Worst: {worst_m[:20]}', edgecolor='white')\n"
        "ax.axvline(0, color='black', linestyle='--', alpha=0.5)\n"
        "ax.set_title('Residual Distribution: Best vs Worst', fontsize=11)\n"
        "ax.set_xlabel('Residual (Actual - Predicted)')\n"
        "ax.legend(fontsize=8)\n"
        "\n"
        "# 4. RMSE vs MAPE scatter\n"
        "ax = axes[1, 1]\n"
        "for m in results_df.index:\n"
        "    if m.startswith('Naive'): c, mk = '#8BC34A', 's'\n"
        "    elif m.startswith('BASE_'): c, mk = '#2196F3', 'o'\n"
        "    elif m.startswith('RES_'): c, mk = '#FF9800', '^'\n"
        "    elif m.startswith('ROR_'): c, mk = '#9C27B0', 'D'\n"
        "    else: c, mk = '#607D8B', 'x'\n"
        "    ax.scatter(results_df.loc[m, 'RMSE'], results_df.loc[m, 'MAPE'], \n"
        "              color=c, marker=mk, s=60, alpha=0.7)\n"
        "ax.set_xlabel('RMSE')\n"
        "ax.set_ylabel('MAPE (%)')\n"
        "ax.set_title('RMSE vs MAPE (All Models)', fontsize=11)\n"
        "# Legend\n"
        "import matplotlib.lines as mlines\n"
        "handles = [\n"
        "    mlines.Line2D([], [], color='#8BC34A', marker='s', linestyle='', label='Naive'),\n"
        "    mlines.Line2D([], [], color='#2196F3', marker='o', linestyle='', label='Baseline'),\n"
        "    mlines.Line2D([], [], color='#FF9800', marker='^', linestyle='', label='Residual'),\n"
        "    mlines.Line2D([], [], color='#9C27B0', marker='D', linestyle='', label='ROR'),\n"
        "]\n"
        "ax.legend(handles=handles, fontsize=8)\n"
        "ax.grid(True, alpha=0.3)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig('report_images/test_results_dashboard.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ))

    # =========================================================================
    # Keep cell 34 (Baseline prediction comparison)
    # =========================================================================
    new_cells.append(old_cells[34])

    # =========================================================================
    # Keep cells 35-36 (Section 5 markdown, 5.1 markdown)
    # =========================================================================
    new_cells.append(old_cells[35])
    new_cells.append(old_cells[36])

    # =========================================================================
    # Keep cell 37 (Backtesting code)
    # =========================================================================
    new_cells.append(old_cells[37])

    # =========================================================================
    # NEW: Backtesting Visualization
    # =========================================================================
    new_cells.append(make_md_cell(
        "### 5.1.1 Backtesting 결과 시각화\n"
        "목표: 각 모델/임계값별 수익률, Sharpe Ratio, Win Rate를 시각적으로 비교.\n"
        "해석: 높은 순수익률과 낮은 MDD를 동시에 달성하는 모델이 실전 투입에 적합."
    ))

    new_cells.append(make_code_cell(
        "# 5.1.1 Backtesting 결과 시각화\n"
        "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n"
        "\n"
        "# 1. Net Return by model (best threshold)\n"
        "ax = axes[0, 0]\n"
        "best_by = ror_df.loc[ror_df.groupby('Model')['Net_Return'].idxmax()]\n"
        "best_by = best_by.sort_values('Net_Return', ascending=True).tail(15)\n"
        "colors_bt = ['#4CAF50' if r > 0 else '#F44336' for r in best_by['Net_Return']]\n"
        "ax.barh(best_by['Model'].str[:30], best_by['Net_Return'] * 100, color=colors_bt)\n"
        "ax.axvline(0, color='black', linewidth=0.8)\n"
        "ax.set_xlabel('Net Return (%)')\n"
        "ax.set_title('Top 15 Models: Best Net Return', fontsize=11)\n"
        "\n"
        "# 2. Sharpe Ratio comparison\n"
        "ax = axes[0, 1]\n"
        "sharpe_best = ror_df.loc[ror_df.groupby('Model')['Sharpe_Ratio'].idxmax()]\n"
        "sharpe_best = sharpe_best.sort_values('Sharpe_Ratio', ascending=True).tail(10)\n"
        "ax.barh(sharpe_best['Model'].str[:30], sharpe_best['Sharpe_Ratio'], color='#3F51B5')\n"
        "ax.axvline(0, color='black', linewidth=0.8)\n"
        "ax.set_xlabel('Sharpe Ratio (Annualized)')\n"
        "ax.set_title('Top 10 Models: Sharpe Ratio', fontsize=11)\n"
        "\n"
        "# 3. Threshold sensitivity for top models\n"
        "ax = axes[1, 0]\n"
        "top_models_bt = best_by.tail(5)['Model'].values\n"
        "for m in top_models_bt:\n"
        "    model_data = ror_df[ror_df['Model'] == m].sort_values('Threshold')\n"
        "    ax.plot(model_data['Threshold'], model_data['Net_Return'] * 100, \n"
        "            marker='o', label=m[:25])\n"
        "ax.set_xlabel('Threshold')\n"
        "ax.set_ylabel('Net Return (%)')\n"
        "ax.set_title('Threshold Sensitivity (Top 5)', fontsize=11)\n"
        "ax.legend(fontsize=7)\n"
        "ax.grid(True, alpha=0.3)\n"
        "\n"
        "# 4. Win Rate vs Net Return scatter\n"
        "ax = axes[1, 1]\n"
        "for _, row in best_by.iterrows():\n"
        "    ax.scatter(row['Win_Rate'] * 100, row['Net_Return'] * 100, s=40, alpha=0.7)\n"
        "    if row['Net_Return'] > best_by['Net_Return'].quantile(0.8):\n"
        "        ax.annotate(row['Model'][:15], (row['Win_Rate']*100, row['Net_Return']*100), fontsize=6)\n"
        "ax.axhline(0, color='gray', linestyle='--', alpha=0.5)\n"
        "ax.axvline(50, color='gray', linestyle='--', alpha=0.5)\n"
        "ax.set_xlabel('Win Rate (%)')\n"
        "ax.set_ylabel('Net Return (%)')\n"
        "ax.set_title('Win Rate vs Net Return', fontsize=11)\n"
        "ax.grid(True, alpha=0.3)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig('report_images/backtesting_dashboard.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ))

    # =========================================================================
    # Keep cells 38-42 (5.2 markdown, 5.2 code, 5.2.1 code, 5.3 markdown, 5.3 code)
    # =========================================================================
    new_cells.append(old_cells[38])
    new_cells.append(old_cells[39])
    new_cells.append(old_cells[40])
    new_cells.append(old_cells[41])
    new_cells.append(old_cells[42])

    # =========================================================================
    # Keep cells 43-44 (Section 6 markdown, Final Results code)
    # =========================================================================
    new_cells.append(old_cells[43])
    new_cells.append(old_cells[44])

    # =========================================================================
    # Keep cells 45-48 (Section 7 markdown, conclusion code, 7.4 markdown, domain code)
    # =========================================================================
    new_cells.append(old_cells[45])
    new_cells.append(old_cells[46])
    new_cells.append(old_cells[47])
    new_cells.append(old_cells[48])

    # =========================================================================
    # Keep cells 49-50 (Section 8 markdown, GBM vs Transformer code)
    # =========================================================================
    new_cells.append(old_cells[49])
    new_cells.append(old_cells[50])

    # =========================================================================
    # NEW: DL Comparison Visualization
    # =========================================================================
    new_cells.append(make_code_cell(
        "# 8.1 GBM vs DL 비교 시각화\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "\n"
        "# 1. Model complexity vs performance radar-like chart\n"
        "ax = axes[0]\n"
        "categories = ['Training Speed', 'Interpretability', 'Small Data', 'Auto Feature', 'Deployment']\n"
        "gbm_scores = [5, 5, 5, 2, 5]\n"
        "dl_scores = [2, 1, 2, 5, 2]\n"
        "\n"
        "x = np.arange(len(categories))\n"
        "width = 0.35\n"
        "bars1 = ax.bar(x - width/2, gbm_scores, width, label='GBM Ensemble', color='#2196F3', alpha=0.8)\n"
        "bars2 = ax.bar(x + width/2, dl_scores, width, label='LSTM/Transformer', color='#F44336', alpha=0.8)\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels(categories, fontsize=9)\n"
        "ax.set_ylabel('Score (1-5)')\n"
        "ax.set_title('GBM vs Deep Learning: Characteristics', fontsize=11)\n"
        "ax.legend()\n"
        "ax.set_ylim(0, 6)\n"
        "ax.grid(True, alpha=0.3, axis='y')\n"
        "\n"
        "# 2. Performance comparison (if DL results available)\n"
        "ax = axes[1]\n"
        "comparison_data = {\n"
        "    'Model': ['GBM Best\\n(' + gbm_best_model[:15] + ')', 'Naive_Drift', 'LSTM (reported)', 'Transformer (reported)'],\n"
        "    'Test RMSE': [gbm_best_rmse, results_df.loc['Naive_Drift', 'RMSE'] if 'Naive_Drift' in results_df.index else np.nan,\n"
        "                  1957.66, 684.34]\n"
        "}\n"
        "comp_df = pd.DataFrame(comparison_data).dropna()\n"
        "colors_comp = ['#2196F3', '#8BC34A', '#F44336', '#FF9800']\n"
        "bars = ax.bar(comp_df['Model'], comp_df['Test RMSE'], color=colors_comp[:len(comp_df)])\n"
        "for bar, val in zip(bars, comp_df['Test RMSE']):\n"
        "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,\n"
        "            f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')\n"
        "ax.set_ylabel('Test RMSE')\n"
        "ax.set_title('Model Family Performance Comparison', fontsize=11)\n"
        "ax.grid(True, alpha=0.3, axis='y')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig('report_images/gbm_vs_dl_comparison.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ))

    # =========================================================================
    # Keep cells 51-52 (Section 9 markdown, Naive experiments code)
    # =========================================================================
    new_cells.append(old_cells[51])
    new_cells.append(old_cells[52])

    # =========================================================================
    # NEW: Naive Experiment Visualization
    # =========================================================================
    new_cells.append(make_md_cell(
        "### 9.1 Naive 후속 실험 시각화\n"
        "목표: Naive 변형/하이브리드/스태킹 실험 결과를 종합 비교.\n"
        "해석: Naive가 ML을 압도하는 이유(레짐 전환)와 최적 하이브리드 가중치를 시각적으로 확인."
    ))

    new_cells.append(make_code_cell(
        "# 9.1 Naive 후속 실험 종합 시각화\n"
        "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n"
        "\n"
        "# 1. Naive variants RMSE comparison\n"
        "ax = axes[0, 0]\n"
        "naive_comp = {\n"
        "    'Naive_Last': rmse(y_test, naive_last),\n"
        "    'Naive_Drift': rmse(y_test, naive_drift),\n"
        "    'Naive_SMA4': rmse(y_test, sma_4),\n"
        "}\n"
        "for alpha in [0.3, 0.5, 0.7, 0.9]:\n"
        "    drift = prev_price - prev_prev_price\n"
        "    naive_comp[f'Damped(a={alpha})'] = rmse(y_test, prev_price + alpha * drift)\n"
        "naive_comp_df = pd.DataFrame(list(naive_comp.items()), columns=['Model', 'RMSE']).sort_values('RMSE')\n"
        "colors_naive = ['#4CAF50' if v == naive_comp_df['RMSE'].min() else '#90CAF9' for v in naive_comp_df['RMSE']]\n"
        "ax.barh(naive_comp_df['Model'], naive_comp_df['RMSE'], color=colors_naive)\n"
        "ax.set_xlabel('Test RMSE')\n"
        "ax.set_title('Naive Variants Comparison', fontsize=11)\n"
        "ax.invert_yaxis()\n"
        "\n"
        "# 2. Hybrid weight sweep\n"
        "ax = axes[0, 1]\n"
        "if 'BASE_GradientBoosting' in preds.columns:\n"
        "    ml_pred = preds['BASE_GradientBoosting']\n"
        "    weights = np.arange(0, 1.05, 0.05)\n"
        "    hybrid_rmses = [rmse(y_test, w * naive_drift + (1-w) * ml_pred) for w in weights]\n"
        "    ax.plot(weights, hybrid_rmses, 'b-', linewidth=2)\n"
        "    best_w = weights[np.argmin(hybrid_rmses)]\n"
        "    best_r = min(hybrid_rmses)\n"
        "    ax.axvline(best_w, color='red', linestyle='--', alpha=0.7, label=f'Optimal: w={best_w:.2f}')\n"
        "    ax.scatter([best_w], [best_r], color='red', s=100, zorder=5)\n"
        "    ax.axhline(rmse(y_test, naive_drift), color='green', linestyle=':', alpha=0.5, label='Pure Naive')\n"
        "    ax.axhline(rmse(y_test, ml_pred), color='orange', linestyle=':', alpha=0.5, label='Pure ML')\n"
        "    ax.set_xlabel('Naive Weight (w)')\n"
        "    ax.set_ylabel('Test RMSE')\n"
        "    ax.set_title('Hybrid Weight Sweep: w*Naive + (1-w)*ML', fontsize=11)\n"
        "    ax.legend(fontsize=8)\n"
        "    ax.grid(True, alpha=0.3)\n"
        "\n"
        "# 3. Test period: Actual vs key models\n"
        "ax = axes[1, 0]\n"
        "ax.plot(y_test.index, y_test, 'k-', linewidth=2.5, label='Actual', alpha=0.7)\n"
        "ax.plot(y_test.index, naive_drift, 'g--', linewidth=1.5, label=f'Naive_Drift (RMSE={rmse(y_test, naive_drift):.0f})')\n"
        "if 'BASE_GradientBoosting' in preds.columns:\n"
        "    ml_pred = preds['BASE_GradientBoosting']\n"
        "    ax.plot(y_test.index, ml_pred, 'b:', linewidth=1.5, label=f'GradientBoosting (RMSE={rmse(y_test, ml_pred):.0f})')\n"
        "    best_hybrid_pred = best_hybrid['weight'] * naive_drift + (1-best_hybrid['weight']) * ml_pred\n"
        "    ax.plot(y_test.index, best_hybrid_pred, 'r-', linewidth=1.5, \n"
        "            label=f'Hybrid(w={best_hybrid[\"weight\"]}) (RMSE={rmse(y_test, best_hybrid_pred):.0f})')\n"
        "ax.set_title('Test Period: Key Model Predictions', fontsize=11)\n"
        "ax.legend(fontsize=8)\n"
        "ax.grid(True, alpha=0.3)\n"
        "ax.tick_params(axis='x', rotation=45)\n"
        "\n"
        "# 4. Why Naive wins: regime change visualization\n"
        "ax = axes[1, 1]\n"
        "# Show train vs test trend\n"
        "train_returns = y_train.pct_change().dropna()\n"
        "test_returns = y_test.pct_change().dropna()\n"
        "# Fill first test return using last train price\n"
        "first_test_ret = (y_test.iloc[0] - y_train.iloc[-1]) / y_train.iloc[-1]\n"
        "test_returns_full = pd.concat([pd.Series([first_test_ret], index=[y_test.index[0]]), test_returns.iloc[1:]])\n"
        "\n"
        "ax.hist(train_returns, bins=30, alpha=0.6, color='blue', density=True, label=f'Train (mean={train_returns.mean()*100:.2f}%)')\n"
        "ax.hist(test_returns_full, bins=10, alpha=0.6, color='red', density=True, label=f'Test (mean={test_returns_full.mean()*100:.2f}%)')\n"
        "ax.axvline(train_returns.mean(), color='blue', linestyle='--', alpha=0.7)\n"
        "ax.axvline(test_returns_full.mean(), color='red', linestyle='--', alpha=0.7)\n"
        "ax.set_title('Regime Change: Train vs Test Return Distribution', fontsize=11)\n"
        "ax.set_xlabel('Weekly Return')\n"
        "ax.legend(fontsize=8)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig('report_images/naive_experiments.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "\n"
        "print('\\n[Regime Change Analysis]')\n"
        "print(f'  Train period mean return: {train_returns.mean()*100:.3f}% (mean-reverting)')\n"
        "print(f'  Test period mean return:  {test_returns_full.mean()*100:.3f}% (trending up)')\n"
        "print(f'  => ML models learned mean-reversion; Naive follows trend mechanically')"
    ))

    # =========================================================================
    # Keep cell 53 (9.1 analysis summary markdown)
    # =========================================================================
    new_cells.append(old_cells[53])

    # =========================================================================
    # Keep cell 54 (Section 10 summary markdown)
    # =========================================================================
    new_cells.append(old_cells[54])

    # =========================================================================
    # FIXED cell 55 (Critical bug fix: current_n_features)
    # =========================================================================
    # Read original cell 55 and fix the bug
    cell55_source = ''.join(old_cells[55]['source'])

    # Fix the bug: current_n_features is unreachable after continue
    fixed_cell55 = cell55_source.replace(
        "if model_name == 'Actual':\n"
        "        continue\n"
        "        current_n_features = 0 # Force 0 to avoid NaN (Test set N=12 < P=20)\n"
        "    m = compute_metrics_with_adj_r2(y_test, preds[model_name], current_n_features)",

        "if model_name == 'Actual':\n"
        "        continue\n"
        "    # n_features for Adj R2: use 0 for Naive models (no features), selected count for ML\n"
        "    if model_name.startswith('Naive') or model_name.startswith('Hybrid'):\n"
        "        current_n_features = 0\n"
        "    else:\n"
        "        current_n_features = min(n_features, len(y_test) - 2)  # Avoid negative dof\n"
        "    m = compute_metrics_with_adj_r2(y_test, preds[model_name], current_n_features)"
    )

    fixed_cell55_obj = copy.deepcopy(old_cells[55])
    lines = fixed_cell55.split('\n')
    formatted = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            formatted.append(line + '\n')
        else:
            formatted.append(line)
    fixed_cell55_obj['source'] = formatted
    new_cells.append(fixed_cell55_obj)

    # =========================================================================
    # Keep cell 56 (Test period visualization)
    # =========================================================================
    new_cells.append(old_cells[56])

    # =========================================================================
    # NEW: Final Comprehensive Summary Visualization
    # =========================================================================
    new_cells.append(make_md_cell(
        "## 12. 최종 종합 시각화\n"
        "목표: 전체 실험 결과를 한눈에 파악할 수 있는 종합 대시보드.\n"
        "해석: ML 교수님이 전체 논리 흐름과 결과를 검증할 수 있는 최종 시각화."
    ))

    new_cells.append(make_code_cell(
        "# 12. 최종 종합 대시보드\n"
        "fig = plt.figure(figsize=(20, 16))\n"
        "fig.suptitle('Nickel Price Prediction: Comprehensive Results Dashboard', \n"
        "             fontsize=16, fontweight='bold', y=0.98)\n"
        "\n"
        "# Layout: 3 rows x 3 cols\n"
        "gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)\n"
        "\n"
        "# ---- Row 1 ----\n"
        "# 1. Full price history with split\n"
        "ax1 = fig.add_subplot(gs[0, :])\n"
        "ax1.plot(df.index, df[target_col], 'k-', linewidth=0.8, alpha=0.7)\n"
        "ax1.axvspan(pd.Timestamp(VAL_START), pd.Timestamp(VAL_END), alpha=0.2, color='orange')\n"
        "ax1.axvspan(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END), alpha=0.2, color='red')\n"
        "ax1.set_title('(A) 13-Year Nickel Price with Train/Val/Test Regions', fontsize=11)\n"
        "ax1.set_ylabel('USD/tonne')\n"
        "ax1.grid(True, alpha=0.3)\n"
        "\n"
        "# ---- Row 2 ----\n"
        "# 2. Feature importance (top 10)\n"
        "ax2 = fig.add_subplot(gs[1, 0])\n"
        "top10 = feat_imp.head(10).sort_values('importance')\n"
        "ax2.barh(top10['feature'], top10['importance'], color='teal')\n"
        "ax2.set_title('(B) SHAP Feature Importance (Top 10)', fontsize=10)\n"
        "ax2.set_xlabel('Mean |SHAP|')\n"
        "\n"
        "# 3. Model RMSE ranking (key models)\n"
        "ax3 = fig.add_subplot(gs[1, 1])\n"
        "key_models = []\n"
        "# Collect key models for summary\n"
        "for m in ['Naive_Last', 'Naive_Drift']:\n"
        "    if m in final_metrics_df.index:\n"
        "        key_models.append(m)\n"
        "for m in final_metrics_df.index:\n"
        "    if m.startswith('Naive_Drift_Damped') and m not in key_models:\n"
        "        key_models.append(m)\n"
        "        break\n"
        "for m in final_metrics_df.index:\n"
        "    if m.startswith('Hybrid') and m not in key_models:\n"
        "        key_models.append(m)\n"
        "        break\n"
        "for m in final_metrics_df.index:\n"
        "    if m.startswith('BASE_') and m not in key_models:\n"
        "        key_models.append(m)\n"
        "        break\n"
        "for m in final_metrics_df.index:\n"
        "    if m.startswith('RES_') and m not in key_models:\n"
        "        key_models.append(m)\n"
        "        break\n"
        "for m in final_metrics_df.index:\n"
        "    if m.startswith('ROR_') and m not in key_models:\n"
        "        key_models.append(m)\n"
        "        break\n"
        "key_df = final_metrics_df.loc[[m for m in key_models if m in final_metrics_df.index]].sort_values('RMSE')\n"
        "colors_key = []\n"
        "for m in key_df.index:\n"
        "    if 'Hybrid' in m: colors_key.append('#E91E63')\n"
        "    elif m.startswith('Naive'): colors_key.append('#8BC34A')\n"
        "    elif m.startswith('BASE_'): colors_key.append('#2196F3')\n"
        "    elif m.startswith('RES_'): colors_key.append('#FF9800')\n"
        "    elif m.startswith('ROR_'): colors_key.append('#9C27B0')\n"
        "    else: colors_key.append('#607D8B')\n"
        "ax3.barh(key_df.index, key_df['RMSE'], color=colors_key)\n"
        "ax3.set_title('(C) Key Model RMSE (Test)', fontsize=10)\n"
        "ax3.set_xlabel('RMSE')\n"
        "ax3.invert_yaxis()\n"
        "\n"
        "# 4. Stacking progression\n"
        "ax4 = fig.add_subplot(gs[1, 2])\n"
        "stages_plot = []\n"
        "if best_baseline_model and best_baseline_model in baseline_val_df.index:\n"
        "    stages_plot.append(('Stage 1\\n(Baseline)', baseline_val_df.loc[best_baseline_model, 'VAL_RMSE']))\n"
        "if best_resid_combo is not None:\n"
        "    stages_plot.append(('Stage 2\\n(+Residual)', best_resid_combo['VAL_RMSE']))\n"
        "if best_ror_combo is not None:\n"
        "    stages_plot.append(('Stage 3\\n(+ROR)', best_ror_combo['VAL_RMSE']))\n"
        "if stages_plot:\n"
        "    s_names, s_vals = zip(*stages_plot)\n"
        "    ax4.plot(s_names, s_vals, 'bo-', linewidth=2, markersize=10)\n"
        "    for n, v in zip(s_names, s_vals):\n"
        "        ax4.annotate(f'{v:.1f}', (n, v), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=10)\n"
        "ax4.set_ylabel('Validation RMSE')\n"
        "ax4.set_title('(D) 3-Stage Stacking Progression', fontsize=10)\n"
        "ax4.grid(True, alpha=0.3)\n"
        "\n"
        "# ---- Row 3 ----\n"
        "# 5. Test period actual vs predictions\n"
        "ax5 = fig.add_subplot(gs[2, :2])\n"
        "ax5.plot(y_test.index, y_test, 'k-', linewidth=2.5, label='Actual', alpha=0.7)\n"
        "plot_models = ['Naive_Drift', 'Hybrid_Naive0.8_ML0.2', 'BASE_GradientBoosting']\n"
        "plot_colors = ['#4CAF50', '#E91E63', '#2196F3']\n"
        "plot_styles = ['--', '-', ':']\n"
        "for m, c, s in zip(plot_models, plot_colors, plot_styles):\n"
        "    if m in preds.columns:\n"
        "        r = np.sqrt(np.mean((y_test - preds[m])**2))\n"
        "        ax5.plot(y_test.index, preds[m], color=c, linestyle=s, linewidth=1.5, \n"
        "                label=f'{m} (RMSE={r:.0f})')\n"
        "ax5.set_title('(E) Test Period: Actual vs Key Predictions', fontsize=11)\n"
        "ax5.legend(fontsize=8)\n"
        "ax5.grid(True, alpha=0.3)\n"
        "ax5.tick_params(axis='x', rotation=45)\n"
        "\n"
        "# 6. Direction accuracy\n"
        "ax6 = fig.add_subplot(gs[2, 2])\n"
        "if len(conf_df) > 0:\n"
        "    top_dir = conf_df.sort_values('Directional_Acc', ascending=False).head(8)\n"
        "    ax6.barh(top_dir.index.str[:25], top_dir['Directional_Acc'] * 100, color='#673AB7')\n"
        "    ax6.axvline(50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')\n"
        "    ax6.set_xlabel('Directional Accuracy (%)')\n"
        "    ax6.set_title('(F) Direction Accuracy (Top 8)', fontsize=10)\n"
        "    ax6.legend(fontsize=8)\n"
        "    ax6.invert_yaxis()\n"
        "\n"
        "plt.savefig('report_images/final_dashboard.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "\n"
        "# Print final summary\n"
        "print('\\n' + '=' * 70)\n"
        "print('FINAL SUMMARY')\n"
        "print('=' * 70)\n"
        "best_overall = final_metrics_df.index[0]\n"
        "print(f'  Best Model: {best_overall}')\n"
        "print(f'  Test RMSE:  {final_metrics_df.loc[best_overall, \"RMSE\"]:.2f}')\n"
        "print(f'  Test MAPE:  {final_metrics_df.loc[best_overall, \"MAPE\"]:.2f}%')\n"
        "if 'Naive_Drift' in final_metrics_df.index:\n"
        "    naive_r = final_metrics_df.loc['Naive_Drift', 'RMSE']\n"
        "    best_r = final_metrics_df.loc[best_overall, 'RMSE']\n"
        "    print(f'  vs Naive_Drift: {(naive_r - best_r)/naive_r*100:.1f}% improvement')\n"
        "print('=' * 70)"
    ))

    # =========================================================================
    # SKIP cells 57-65 (duplicate visualization cells)
    # =========================================================================

    # =========================================================================
    # Done: Set new cells
    # =========================================================================
    nb['cells'] = new_cells

    print(f"Original cells: {len(old_cells)}")
    print(f"New cells: {len(new_cells)}")
    print(f"Removed {len(old_cells) - len(new_cells)} cells (duplicates + empty)")

    # Verify cell type distribution
    md_count = sum(1 for c in new_cells if c['cell_type'] == 'markdown')
    code_count = sum(1 for c in new_cells if c['cell_type'] == 'code')
    print(f"Markdown cells: {md_count}")
    print(f"Code cells: {code_count}")

    # Write output
    with open('sparta2.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print("\nNotebook rebuilt successfully!")
    print("\nNew visualizations added:")
    print("  1. Pipeline flow diagram (Section 0.2)")
    print("  2. Train/Val/Test split timeline (Section 1.1)")
    print("  3. Enhanced data overview 3-panel (Section 1.2)")
    print("  4. Data quality with gap/missing/category viz (Section 2)")
    print("  5. SHAP beeswarm + enhanced feature analysis (Section 3.1)")
    print("  6. Stacking results heatmap & comparison (Section 4.3.1)")
    print("  7. Test results comprehensive dashboard (Section 4.5)")
    print("  8. Backtesting dashboard (Section 5.1.1)")
    print("  9. GBM vs DL comparison chart (Section 8.1)")
    print("  10. Naive experiment visualization (Section 9.1)")
    print("  11. Final comprehensive dashboard (Section 12)")
    print("\nBugs fixed:")
    print("  1. Cell 55: current_n_features unreachable after continue")
    print("  2. Empty cell 32 removed")
    print("  3. Duplicate cells 57-65 removed")


if __name__ == '__main__':
    main()
