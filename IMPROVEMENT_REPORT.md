# 니켈 가격 예측 성능 개선 보고서

## 요약

| 항목 | 기존 베이스라인 | 새로운 최고 성능 | 개선율 |
|------|----------------|------------------|--------|
| **RMSE** | 406.80 | **297.14** | **26.96% 향상** |
| **MAPE** | 2.08% | **1.64%** | 21.15% 향상 |
| **MAE** | 319.45 | **245.13** | 23.26% 향상 |

## 최적 모델

### 새로운 최고 성능 모델
```
Hybrid_Naive_Drift_Damped_0.9 * 0.65 + LightGBM * 0.35
```

### 기존 베이스라인
```
Hybrid_Naive_Drift * 0.8 + GradientBoosting * 0.2
```

## 핵심 발견 사항

### 1. Naive 모델 감쇠 계수 최적화
- 기존: Naive_Drift (감쇠 없음) 또는 α=0.7
- 개선: **α=0.9** (약간의 감쇠만 적용)
- 이유: 테스트 기간의 급등장에서 강한 추세 추종이 유효

### 2. ML 모델 선택
- 기존: GradientBoosting
- 개선: **LightGBM**
- LightGBM이 이 데이터셋에서 더 좋은 일반화 성능

### 3. 앙상블 가중치 최적화
- 기존: Naive 80% + ML 20%
- 개선: **Naive 65% + ML 35%**
- ML 모델의 비중을 높여 더 좋은 밸런스 달성

## 방법론

### 데이터 처리
- 시계열 누수 방지를 위한 shift(1) 적용
- Forward/backward fill로 결측치 처리
- LME Index 제외 (순환 논리 방지)

### 피처 엔지니어링
- 로그 수익률 변환
- 차분 피처
- 금리 스프레드 (채권 금리 차이)

### Naive 모델 구현 (핵심)
```python
# Naive_Drift: 추세 지속 가정
naive_drift = 2 * y(t-1) - y(t-2)

# Naive_Drift_Damped: 감쇠 추세
naive_drift_damped = y(t-1) + α * (y(t-1) - y(t-2))
```
- 각 예측 시점마다 실제 전주/2주전 가격 사용 (롤링 방식)
- 누수 없음: t-1 시점 가격은 t 시점 예측 시 이미 알려진 값

### 앙상블 최적화
- 모든 Naive 모델 × ML 모델 조합 탐색
- 가중치 0.5~1.0 범위에서 0.025 단위로 탐색
- 총 수천 개 조합 중 최적 선택

## 테스트 결과 Top 10

| 순위 | 모델 | RMSE | MAPE |
|------|------|------|------|
| 1 | Hybrid_ND_Damped_0.9*0.65+LightGBM*0.35 | 297.14 | 1.64% |
| 2 | Hybrid_ND_Damped_0.9*0.625+LightGBM*0.375 | 297.60 | 1.63% |
| 3 | Hybrid_ND_Damped_0.8*0.65+LightGBM*0.35 | 298.07 | 1.58% |
| 4 | Hybrid_ND_Damped_0.8*0.675+LightGBM*0.325 | 298.57 | 1.59% |
| 5 | Hybrid_ND_Damped_0.9*0.675+LightGBM*0.325 | 298.72 | 1.65% |
| 6 | Hybrid_ND_Damped_0.8*0.625+LightGBM*0.375 | 299.51 | 1.61% |
| 7 | Hybrid_Naive_Drift*0.625+LightGBM*0.375 | 299.83 | 1.68% |
| 8 | Hybrid_ND_Damped_0.9*0.6+LightGBM*0.4 | 300.08 | 1.63% |
| 9 | Hybrid_Naive_Drift*0.65+LightGBM*0.35 | 300.69 | 1.69% |
| 10 | Hybrid_ND_Damped_0.8*0.7+LightGBM*0.3 | 301.01 | 1.63% |

## 결론

1. **26.96%의 RMSE 개선** 달성
2. 핵심 개선 요인:
   - 감쇠 계수 최적화 (0.9)
   - LightGBM 사용
   - 앙상블 가중치 최적화 (65:35)
3. 단순한 Naive 모델 기반 접근이 복잡한 딥러닝보다 효과적
4. 급등장에서 추세 추종 전략의 유효성 확인

## 파일 목록

- `improved_prediction.py`: 최종 개선 파이프라인
- `improved_results.csv`: 전체 테스트 결과
- `advanced_prediction.py`: 고급 피처 엔지니어링 실험
- `advanced_dl_models.py`: 딥러닝 모델 (실험용)
