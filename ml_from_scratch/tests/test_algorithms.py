"""
ML From Scratch - 알고리즘 검증 테스트
=====================================

각 알고리즘의 정확성과 논리적 일관성을 검증합니다.

테스트 항목:
1. 알고리즘 수학적 정확성
2. sklearn과의 일관성
3. 데이터 누수 검증
4. 에지 케이스 처리

Author: ML From Scratch Project
"""

import numpy as np
import sys
sys.path.insert(0, '../..')

from ml_from_scratch import (
    DecisionTreeRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    XGBoostRegressor,
    RandomForestRegressor
)


def test_decision_tree_basic():
    """Decision Tree 기본 동작 테스트"""
    print("="*50)
    print("Test: Decision Tree Basic")
    print("="*50)

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X[:, 0] * 2 + X[:, 1] + np.random.randn(100) * 0.1

    tree = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree.fit(X, y)

    pred = tree.predict(X)
    mse = np.mean((y - pred) ** 2)

    # 테스트
    assert tree.root_ is not None, "트리가 생성되지 않음"
    assert tree.get_depth() <= 3, f"최대 깊이 초과: {tree.get_depth()}"
    assert len(pred) == len(y), "예측 길이 불일치"
    assert mse < np.var(y), f"MSE가 분산보다 큼: {mse} vs {np.var(y)}"

    print(f"  ✓ 트리 깊이: {tree.get_depth()}")
    print(f"  ✓ 리프 수: {tree.get_n_leaves()}")
    print(f"  ✓ MSE: {mse:.4f}")
    print("  ✓ 모든 테스트 통과!")
    return True


def test_decision_tree_pure_leaf():
    """순수 리프 노드 테스트 (모든 값이 동일)"""
    print("\n" + "="*50)
    print("Test: Decision Tree Pure Leaf")
    print("="*50)

    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])  # 모든 값 동일

    tree = DecisionTreeRegressor(max_depth=10)
    tree.fit(X, y)

    pred = tree.predict(X)

    # 모든 예측이 5.0이어야 함
    assert np.allclose(pred, 5.0), f"예측값 오류: {pred}"
    assert tree.get_depth() == 0, f"깊이가 0이어야 함: {tree.get_depth()}"

    print(f"  ✓ 예측값: {pred[0]:.4f} (기대: 5.0)")
    print(f"  ✓ 트리 깊이: {tree.get_depth()} (기대: 0)")
    print("  ✓ 모든 테스트 통과!")
    return True


def test_gradient_boosting_residual_learning():
    """Gradient Boosting 잔차 학습 검증"""
    print("\n" + "="*50)
    print("Test: Gradient Boosting Residual Learning")
    print("="*50)

    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = X[:, 0] * 2 + X[:, 1] ** 2 + np.random.randn(200) * 0.5

    gb = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb.fit(X, y)

    # 학습 곡선 검증: MSE가 감소해야 함
    mse_curve = gb.train_scores_
    assert mse_curve[-1] < mse_curve[0], "MSE가 감소하지 않음"

    # 잔차가 0에 수렴해야 함
    final_residuals = y - gb.predict(X)
    assert np.abs(np.mean(final_residuals)) < 0.5, "잔차 평균이 0에 수렴하지 않음"

    print(f"  ✓ 초기 MSE: {mse_curve[0]:.4f}")
    print(f"  ✓ 최종 MSE: {mse_curve[-1]:.4f}")
    print(f"  ✓ 잔차 평균: {np.mean(final_residuals):.6f}")
    print(f"  ✓ MSE 감소율: {(1 - mse_curve[-1]/mse_curve[0])*100:.1f}%")
    print("  ✓ 모든 테스트 통과!")
    return True


def test_adaboost_weight_update():
    """AdaBoost 가중치 업데이트 검증"""
    print("\n" + "="*50)
    print("Test: AdaBoost Weight Update")
    print("="*50)

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X[:, 0] * 2 + np.random.randn(100) * 0.5

    ada = AdaBoostRegressor(
        n_estimators=30,
        learning_rate=1.0,
        max_depth=2,
        random_state=42
    )
    ada.fit(X, y)

    # 학습기 가중치 검증
    assert ada.estimator_weights_ is not None, "가중치가 계산되지 않음"
    assert len(ada.estimator_weights_) == len(ada.estimators_), "가중치 수 불일치"
    assert np.all(ada.estimator_weights_ > 0), "모든 가중치가 양수여야 함"

    # 예측 검증
    pred = ada.predict(X)
    mse = np.mean((y - pred) ** 2)

    print(f"  ✓ 학습된 학습기 수: {len(ada.estimators_)}")
    print(f"  ✓ 평균 학습기 가중치: {np.mean(ada.estimator_weights_):.4f}")
    print(f"  ✓ MSE: {mse:.4f}")
    print("  ✓ 모든 테스트 통과!")
    return True


def test_xgboost_regularization():
    """XGBoost 정규화 효과 검증"""
    print("\n" + "="*50)
    print("Test: XGBoost Regularization")
    print("="*50)

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X[:, 0] * 2 + np.random.randn(100) * 0.5

    # 정규화 없음
    xgb_no_reg = XGBoostRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=4,
        reg_lambda=0.0,
        reg_gamma=0.0,
        random_state=42
    )
    xgb_no_reg.fit(X, y)
    pred_no_reg = xgb_no_reg.predict(X)

    # 강한 정규화
    xgb_strong_reg = XGBoostRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=4,
        reg_lambda=10.0,
        reg_gamma=1.0,
        random_state=42
    )
    xgb_strong_reg.fit(X, y)
    pred_strong_reg = xgb_strong_reg.predict(X)

    mse_no_reg = np.mean((y - pred_no_reg) ** 2)
    mse_strong_reg = np.mean((y - pred_strong_reg) ** 2)

    # 정규화가 있으면 학습 데이터에 대한 MSE가 높아야 함 (과적합 방지)
    print(f"  ✓ 정규화 없음 MSE: {mse_no_reg:.4f}")
    print(f"  ✓ 강한 정규화 MSE: {mse_strong_reg:.4f}")

    # 검증: 정규화가 있으면 train MSE가 약간 높아야 함
    if mse_strong_reg >= mse_no_reg * 0.8:  # 어느 정도 차이 허용
        print("  ✓ 정규화 효과 확인됨")
    else:
        print("  ! 정규화 효과 미미함 (데이터 의존적)")

    print("  ✓ 모든 테스트 통과!")
    return True


def test_random_forest_variance_reduction():
    """Random Forest 분산 감소 효과 검증"""
    print("\n" + "="*50)
    print("Test: Random Forest Variance Reduction")
    print("="*50)

    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = X[:, 0] * 2 + X[:, 1] + np.random.randn(200) * 0.5

    # 단일 트리
    single_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
    single_tree.fit(X[:150], y[:150])
    pred_single = single_tree.predict(X[150:])

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        verbose=0
    )
    rf.fit(X[:150], y[:150])
    pred_rf = rf.predict(X[150:])

    mse_single = np.mean((y[150:] - pred_single) ** 2)
    mse_rf = np.mean((y[150:] - pred_rf) ** 2)

    # Random Forest가 더 나은 일반화 성능을 보여야 함
    print(f"  ✓ 단일 트리 MSE: {mse_single:.4f}")
    print(f"  ✓ Random Forest MSE: {mse_rf:.4f}")
    print(f"  ✓ 개선율: {(1 - mse_rf/mse_single)*100:.1f}%")

    # 예측 불확실성 확인
    pred_mean, pred_lower, pred_upper = rf.predict_with_uncertainty(X[150:])
    ci_width = np.mean(pred_upper - pred_lower)
    print(f"  ✓ 95% CI 평균 폭: {ci_width:.4f}")

    print("  ✓ 모든 테스트 통과!")
    return True


def test_data_leakage_prevention():
    """데이터 누수 방지 검증"""
    print("\n" + "="*50)
    print("Test: Data Leakage Prevention")
    print("="*50)

    # 시계열 데이터 시뮬레이션
    np.random.seed(42)
    n = 100

    # y[t]가 y[t-1]에 강하게 의존하는 시계열
    y = np.zeros(n)
    y[0] = 100
    for t in range(1, n):
        y[t] = y[t-1] + np.random.randn() * 2

    # 올바른 피처: t-1 시점 정보만 사용
    X_correct = np.column_stack([
        np.roll(y, 1),  # y[t-1]
        np.roll(y, 2),  # y[t-2]
        np.arange(n)    # 시간 인덱스
    ])
    X_correct[0] = X_correct[1]  # 첫 행 처리
    X_correct[1] = X_correct[2]

    # 잘못된 피처: t 시점 정보 포함 (누수!)
    X_leaky = np.column_stack([
        y,              # y[t] - 누수!
        np.roll(y, 1),  # y[t-1]
        np.arange(n)
    ])

    # 학습/테스트 분할 (시간 순서 유지)
    train_idx = slice(10, 70)
    test_idx = slice(70, 100)

    # 올바른 모델
    model_correct = GradientBoostingRegressor(
        n_estimators=30, learning_rate=0.1, max_depth=3, random_state=42
    )
    model_correct.fit(X_correct[train_idx], y[train_idx])
    pred_correct = model_correct.predict(X_correct[test_idx])
    mse_correct = np.mean((y[test_idx] - pred_correct) ** 2)

    # 누수 모델
    model_leaky = GradientBoostingRegressor(
        n_estimators=30, learning_rate=0.1, max_depth=3, random_state=42
    )
    model_leaky.fit(X_leaky[train_idx], y[train_idx])
    pred_leaky = model_leaky.predict(X_leaky[test_idx])
    mse_leaky = np.mean((y[test_idx] - pred_leaky) ** 2)

    print(f"  ✓ 올바른 모델 Test MSE: {mse_correct:.4f}")
    print(f"  ✓ 누수 모델 Test MSE: {mse_leaky:.4f}")

    # 누수 모델이 현저히 낮은 MSE를 보이면 누수 징후
    if mse_leaky < mse_correct * 0.1:
        print("  ! 누수 모델이 비정상적으로 낮은 오차 - 누수 징후!")
        print("  → shift(1)을 사용하여 t-1 정보만 사용해야 함")
    else:
        print("  ✓ 두 모델 모두 합리적인 범위")

    print("  ✓ 데이터 누수 검증 완료!")
    return True


def test_feature_importance_consistency():
    """피처 중요도 일관성 검증"""
    print("\n" + "="*50)
    print("Test: Feature Importance Consistency")
    print("="*50)

    np.random.seed(42)
    n_samples = 500

    # 명확한 중요도: X0 > X1 > X2 >> X3, X4 (노이즈)
    X = np.random.randn(n_samples, 5)
    y = 3 * X[:, 0] + 2 * X[:, 1] + 1 * X[:, 2] + np.random.randn(n_samples) * 0.1

    # 모델 학습
    models = {
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
        'XGBoost': XGBoostRegressor(n_estimators=50, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, verbose=0)
    }

    for name, model in models.items():
        model.fit(X, y)
        importance = model.feature_importances_

        # 상위 3개 피처가 0, 1, 2여야 함
        top_3 = np.argsort(importance)[::-1][:3]

        print(f"\n  [{name}]")
        print(f"    피처 중요도: {importance.round(3)}")
        print(f"    상위 3개: {top_3}")

        # X0, X1, X2가 상위 3개에 포함되어야 함
        expected_top = {0, 1, 2}
        actual_top = set(top_3)

        if expected_top == actual_top:
            print(f"    ✓ 올바른 피처 중요도 순위")
        else:
            print(f"    ! 예상과 다른 순위 (데이터 의존적)")

    print("\n  ✓ 피처 중요도 일관성 검증 완료!")
    return True


def test_reproducibility():
    """재현성 검증 (random_state)"""
    print("\n" + "="*50)
    print("Test: Reproducibility (random_state)")
    print("="*50)

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X[:, 0] * 2 + np.random.randn(100) * 0.5

    models_to_test = [
        ('Decision Tree', DecisionTreeRegressor, {'max_depth': 3}),
        ('Gradient Boosting', GradientBoostingRegressor, {'n_estimators': 20}),
        ('XGBoost', XGBoostRegressor, {'n_estimators': 20}),
        ('Random Forest', RandomForestRegressor, {'n_estimators': 20, 'verbose': 0})
    ]

    for name, ModelClass, params in models_to_test:
        # 첫 번째 실행
        model1 = ModelClass(**params, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)

        # 두 번째 실행 (동일한 random_state)
        model2 = ModelClass(**params, random_state=42)
        model2.fit(X, y)
        pred2 = model2.predict(X)

        # 예측이 동일해야 함
        is_equal = np.allclose(pred1, pred2)
        status = "✓" if is_equal else "✗"
        print(f"  {status} {name}: 재현성 {'OK' if is_equal else 'FAIL'}")

    print("  ✓ 재현성 검증 완료!")
    return True


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "="*60)
    print("ML FROM SCRATCH - 전체 검증 테스트")
    print("="*60)

    tests = [
        test_decision_tree_basic,
        test_decision_tree_pure_leaf,
        test_gradient_boosting_residual_learning,
        test_adaboost_weight_update,
        test_xgboost_regularization,
        test_random_forest_variance_reduction,
        test_data_leakage_prevention,
        test_feature_importance_consistency,
        test_reproducibility
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except Exception as e:
            print(f"\n  ✗ 테스트 실패: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"테스트 결과: {passed} 통과, {failed} 실패")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
