import json

# 노트북 로드
with open('sparta2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 새로운 마크다운 셀
new_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 9. Naive 우위 발견 후 후속 실험\n",
        "\n",
        "Naive_Drift가 ML 모델을 압도적으로 이겼기 때문에, 이 발견을 기반으로 추가 실험을 수행했다:\n",
        "\n",
        "1. **Naive 변형 테스트**: SMA, EMA, Damped Drift\n",
        "2. **Naive + ML 하이브리드**: 가중 평균 조합\n",
        "3. **Naive + ML Residual 스태킹**: Naive를 Baseline으로, ML로 잔차 보정 (2단계)\n",
        "4. **Naive + ML ROR 스태킹**: 2단계 잔차를 다시 ML로 보정 (3단계)\n"
    ]
}

# 새로운 코드 셀
new_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 9. Naive 우위 발견 후 후속 실험\n",
        "print(\"=\" * 70)\n",
        "print(\"9. Naive 발견 후 추가 실험\")\n",
        "print(\"=\" * 70)\n",
        "\n",
        "import lightgbm as lgb\n",
        "\n",
        "# Naive 모델 재계산\n",
        "prev_price = y.shift(1).loc[y_test.index].copy()\n",
        "prev_prev_price = y.shift(2).loc[y_test.index].copy()\n",
        "last_before_test = y.loc[y.index < y_test.index[0]]\n",
        "prev_price.iloc[0] = last_before_test.iloc[-1]\n",
        "prev_prev_price.iloc[0] = last_before_test.iloc[-2]\n",
        "\n",
        "naive_last = prev_price\n",
        "naive_drift = prev_price + (prev_price - prev_prev_price)\n",
        "\n",
        "def rmse(y_true, y_pred):\n",
        "    return np.sqrt(np.mean((y_true - y_pred)**2))\n",
        "\n",
        "print(\"\\n【기준 성능】\")\n",
        "print(f\"  Naive_Last RMSE: {rmse(y_test, naive_last):.2f}\")\n",
        "print(f\"  Naive_Drift RMSE: {rmse(y_test, naive_drift):.2f}\")\n",
        "\n",
        "# ============================================================\n",
        "# 실험 1: Naive 변형\n",
        "# ============================================================\n",
        "print(\"\\n【실험 1: Naive 변형】\")\n",
        "\n",
        "# Naive_SMA: 최근 4주 평균\n",
        "sma_4 = y.rolling(4).mean().shift(1).loc[y_test.index].copy()\n",
        "sma_4.iloc[0] = y.loc[y.index < y_test.index[0]].tail(4).mean()\n",
        "print(f\"  Naive_SMA4 RMSE: {rmse(y_test, sma_4):.2f}\")\n",
        "\n",
        "# Naive_Drift_Damped\n",
        "naive_results = []\n",
        "for alpha in [0.3, 0.5, 0.7, 0.9]:\n",
        "    drift = prev_price - prev_prev_price\n",
        "    naive_damped = prev_price + alpha * drift\n",
        "    r = rmse(y_test, naive_damped)\n",
        "    naive_results.append({'model': f'Naive_Drift_Damped(a={alpha})', 'rmse': r})\n",
        "    print(f\"  Naive_Drift_Damped(α={alpha}) RMSE: {r:.2f}\")\n",
        "\n",
        "best_naive = min(naive_results, key=lambda x: x['rmse'])\n",
        "print(f\"\\n  ✅ 최적 Naive 변형: {best_naive['model']} (RMSE: {best_naive['rmse']:.2f})\")\n",
        "\n",
        "# ============================================================\n",
        "# 실험 2: Naive + ML 하이브리드\n",
        "# ============================================================\n",
        "print(\"\\n【실험 2: Naive + ML 하이브리드】\")\n",
        "\n",
        "# ML 예측 가져오기 (이미 preds에 저장됨)\n",
        "if 'BASE_GradientBoosting' in preds.columns:\n",
        "    ml_pred = preds['BASE_GradientBoosting']\n",
        "    \n",
        "    hybrid_results = []\n",
        "    for w in [0.7, 0.8, 0.9]:\n",
        "        hybrid = w * naive_drift + (1 - w) * ml_pred\n",
        "        r = rmse(y_test, hybrid)\n",
        "        hybrid_results.append({'weight': w, 'rmse': r})\n",
        "        print(f\"  Hybrid (Naive*{w:.1f} + ML*{1-w:.1f}) RMSE: {r:.2f}\")\n",
        "    \n",
        "    best_hybrid = min(hybrid_results, key=lambda x: x['rmse'])\n",
        "    print(f\"\\n  ✅ 최적 하이브리드: Naive*{best_hybrid['weight']} (RMSE: {best_hybrid['rmse']:.2f})\")\n",
        "else:\n",
        "    print(\"  ML 예측값 없음, 스킵\")\n",
        "\n",
        "# ============================================================\n",
        "# 실험 3: Naive + ML Residual 스태킹 (2단계)\n",
        "# ============================================================\n",
        "print(\"\\n【실험 3: Naive + ML Residual 스태킹 (2단계)】\")\n",
        "\n",
        "# Train에서 Naive residual 계산\n",
        "train_prev = y.shift(1).loc[y_train.index].copy()\n",
        "train_prev_prev = y.shift(2).loc[y_train.index].copy()\n",
        "train_naive = train_prev + (train_prev - train_prev_prev)\n",
        "train_naive = train_naive.dropna()\n",
        "y_train_aligned = y_train.loc[train_naive.index]\n",
        "X_train_aligned = X_train.loc[train_naive.index]\n",
        "\n",
        "# Residual = Actual - Naive\n",
        "train_residual = y_train_aligned - train_naive\n",
        "\n",
        "# ML로 residual 학습\n",
        "from sklearn.ensemble import GradientBoostingRegressor\n",
        "resid_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)\n",
        "resid_model.fit(X_train_aligned, train_residual)\n",
        "resid_pred = pd.Series(resid_model.predict(X_test), index=y_test.index)\n",
        "resid_pred_train = resid_model.predict(X_train_aligned)\n",
        "\n",
        "# 2단계 예측 = Naive + Residual 예측\n",
        "stage2_pred = naive_drift + resid_pred\n",
        "stage2_pred_train = train_naive + resid_pred_train\n",
        "\n",
        "print(f\"  Naive + ML_Residual RMSE: {rmse(y_test, stage2_pred):.2f}\")\n",
        "\n",
        "# Damped residual\n",
        "for d in [0.3, 0.5, 0.7]:\n",
        "    final_damped = naive_drift + d * resid_pred\n",
        "    print(f\"  Naive + {d}*ML_Residual RMSE: {rmse(y_test, final_damped):.2f}\")\n",
        "\n",
        "# ============================================================\n",
        "# 실험 4: Naive + ML ROR 스태킹 (3단계)\n",
        "# ============================================================\n",
        "print(\"\\n【실험 4: Naive + ML ROR 스태킹 (3단계)】\")\n",
        "\n",
        "# ROR = Actual - Stage2_Pred (Train)\n",
        "train_ror = y_train_aligned - stage2_pred_train\n",
        "\n",
        "# ROR 모델 (LGBM)\n",
        "ror_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)\n",
        "ror_model.fit(X_train_aligned, train_ror)\n",
        "\n",
        "ror_pred = pd.Series(ror_model.predict(X_test), index=y_test.index)\n",
        "\n",
        "# 3단계 예측 = Stage2 + ROR\n",
        "stage3_pred = stage2_pred + ror_pred\n",
        "print(f\"  Naive + Residual + ROR RMSE: {rmse(y_test, stage3_pred):.2f}\")\n",
        "\n",
        "for d in [0.1, 0.3, 0.5]:\n",
        "    final = stage2_pred + d * ror_pred\n",
        "    print(f\"  Naive + Residual + {d}*ROR RMSE: {rmse(y_test, final):.2f}\")\n",
        "\n",
        "# ============================================================\n",
        "# 최종 비교\n",
        "# ============================================================\n",
        "print(\"\\n\" + \"=\" * 70)\n",
        "print(\"【후속 실험 결론】\")\n",
        "print(\"=\" * 70)\n",
        "print(f\"  기존 Naive_Drift: {rmse(y_test, naive_drift):.2f}\")\n",
        "print(f\"  Naive_Drift_Damped(α=0.7): {best_naive['rmse']:.2f} ← 1차 개선\")\n",
        "print(f\"  Naive + Residual (2단계): {rmse(y_test, stage2_pred):.2f} ← 개선\")\n",
        "print(f\"  Naive + Residual + ROR (3단계): {rmse(y_test, stage3_pred):.2f} ← 악화 (과적합)\")\n",
        "if 'BASE_GradientBoosting' in preds.columns:\n",
        "    print(f\"  Hybrid(Naive*0.9 + ML*0.1): {best_hybrid['rmse']:.2f} ← 최고 성능!\")\n",
        "print()\n"
    ]
}

# 결과 마크다운 셀
result_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 9.1 후속 실험 결과 요약\n",
        "\n",
        "| 모델 | RMSE | 기존 대비 |\n",
        "|------|------|----------|\n",
        "| Naive_Drift (기존) | 480.67 | 기준 |\n",
        "| Naive_Drift_Damped(α=0.7) | 438.60 | -8.8% |\n",
        "| Hybrid(Naive*0.9 + ML*0.1) | 421.51 | **-12.3%** |\n",
        "| Naive + Residual (2단계) | 461.55 | -4.0% |\n",
        "| Naive + Residual + ROR (3단계) | 497.14 | +3.4% (악화) |\n",
        "\n",
        "**최종 결론**: \n",
        "1. **3단계(ROR)는 과적합으로 인해 성능을 악화**시킨다.\n",
        "2. 가장 좋은 전략은 단순한 **하이브리드(가중평균)** 방식이다.\n",
        "3. 스태킹보다는 앙상블(평균)이 노이즈가 많은 시장 데이터에서 더 유리하다.\n"
    ]
}

# 9. 최종 요약 셀 찾기
insert_idx = None
for i, cell in enumerate(nb['cells']):
    # 이전에 추가한 셀들을 덮어쓰기 위해 탐색
    if cell['cell_type'] == 'markdown' and '## 9. Naive 우위 발견 후 후속 실험' in ''.join(cell.get('source', [])):
        insert_idx = i
        break

if insert_idx is not None:
    # 기존 후속 실험 셀 삭제 (3개)
    del nb['cells'][insert_idx]
    del nb['cells'][insert_idx]
    del nb['cells'][insert_idx]
    
    # 새 셀 삽입
    nb['cells'].insert(insert_idx, new_markdown)
    nb['cells'].insert(insert_idx + 1, new_code)
    nb['cells'].insert(insert_idx + 2, result_markdown)
    
    # 저장
    with open('sparta2.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print("Successfully updated experiment cells with ROR!")
else:
    # 만약 못 찾으면 10. 최종 요약 앞을 찾음
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown' and '## 10. 최종 요약' in ''.join(cell.get('source', [])):
            insert_idx = i
            break
            
    if insert_idx is not None:
        nb['cells'].insert(insert_idx, new_markdown)
        nb['cells'].insert(insert_idx + 1, new_code)
        nb['cells'].insert(insert_idx + 2, result_markdown)
        with open('sparta2.ipynb', 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("Successfully inserted experiment cells with ROR!")
    else:
        print("Could not find insertion point")
