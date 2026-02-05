"""
AdaBoost Regressor - From Scratch Implementation
================================================

AdaBoost.R2 알고리즘 구현 (Drucker 1997).

수학적 배경:
-----------
AdaBoost의 핵심 아이디어:
- 각 샘플에 가중치를 부여
- 예측이 어려운 샘플에 더 높은 가중치
- 다음 학습기가 어려운 샘플에 집중하도록 유도

알고리즘 (AdaBoost.R2):
----------------------
1. 초기화: w_i = 1/n (균등 가중치)

2. for m = 1 to M:
   a. 가중치 w로 학습기 h_m 학습
   b. 각 샘플의 손실 계산:
      L_i = |y_i - h_m(x_i)| / D
      여기서 D = max_i |y_i - h_m(x_i)| (정규화 상수)

   c. 평균 손실 계산:
      L_avg = Σ w_i * L_i

   d. 학습기 가중치 계산:
      β_m = L_avg / (1 - L_avg)

   e. 샘플 가중치 업데이트:
      w_i = w_i * β_m^(1 - L_i)
      w = w / Σw (정규화)

3. 최종 예측 (가중 중앙값):
   - 각 학습기의 예측을 log(1/β_m)로 가중
   - 가중 중앙값 반환

Author: ML From Scratch Project
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from .decision_tree import DecisionTreeRegressor


class AdaBoostRegressor:
    """
    AdaBoost.R2 회귀 모델 (From Scratch)

    Parameters
    ----------
    n_estimators : int, default=50
        부스팅 라운드 수

    learning_rate : float, default=1.0
        학습률 (β를 조절)
        작은 값은 더 보수적인 부스팅

    max_depth : int, default=3
        기본 학습기(트리)의 최대 깊이

    min_samples_split : int, default=2
        내부 노드를 분할하기 위한 최소 샘플 수

    min_samples_leaf : int, default=1
        리프 노드에 있어야 하는 최소 샘플 수

    loss : str, default='linear'
        손실 함수 종류
        - 'linear': L_i = |e_i| / D
        - 'square': L_i = (e_i / D)²
        - 'exponential': L_i = 1 - exp(-|e_i| / D)

    random_state : int, default=None
        랜덤 시드

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        학습된 트리들

    estimator_weights_ : ndarray
        각 학습기의 가중치 log(1/β_m)

    estimator_errors_ : ndarray
        각 학습기의 가중 평균 오차

    feature_importances_ : ndarray
        피처 중요도

    Examples
    --------
    >>> from ml_from_scratch import AdaBoostRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = X[:, 0] * 2 + X[:, 1] + np.random.randn(100) * 0.1
    >>> ada = AdaBoostRegressor(n_estimators=50)
    >>> ada.fit(X, y)
    >>> predictions = ada.predict(X[:5])
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        loss: str = 'linear',
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss
        self.random_state = random_state

        # 학습 후 설정되는 속성들
        self.estimators_: List[DecisionTreeRegressor] = []
        self.estimator_weights_: Optional[np.ndarray] = None
        self.estimator_errors_: Optional[np.ndarray] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.n_features_: int = 0

        # 학습 과정 기록 (시각화용)
        self.training_history_: List[Dict] = []

    def _calculate_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        각 샘플의 손실 계산

        Parameters
        ----------
        y_true : ndarray
            실제 값
        y_pred : ndarray
            예측 값

        Returns
        -------
        loss : ndarray
            각 샘플의 정규화된 손실 (0 ~ 1)
        """
        # 절대 오차 계산
        errors = np.abs(y_true - y_pred)

        # 정규화 상수 (최대 오차)
        D = np.max(errors)

        if D == 0:
            return np.zeros_like(errors)

        # 정규화된 오차
        normalized_errors = errors / D

        # 손실 함수 적용
        if self.loss == 'linear':
            return normalized_errors
        elif self.loss == 'square':
            return normalized_errors ** 2
        elif self.loss == 'exponential':
            return 1 - np.exp(-normalized_errors)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

    def _weighted_median(
        self,
        values: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """
        가중 중앙값 계산

        Parameters
        ----------
        values : ndarray
            값들
        weights : ndarray
            가중치들

        Returns
        -------
        weighted_median : float
            가중 중앙값
        """
        # 정렬
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]

        # 누적 가중치 계산
        cumulative_weights = np.cumsum(sorted_weights)
        total_weight = cumulative_weights[-1]

        # 중앙값 위치 찾기 (누적 가중치가 50%를 넘는 첫 번째 위치)
        median_idx = np.searchsorted(cumulative_weights, total_weight / 2)

        return sorted_values[min(median_idx, len(sorted_values) - 1)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoostRegressor':
        """
        AdaBoost.R2 모델 학습

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            학습 데이터
        y : ndarray of shape (n_samples,)
            타겟 값

        Returns
        -------
        self : AdaBoostRegressor
            학습된 모델
        """
        # 입력 검증
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if X.shape[0] != len(y):
            raise ValueError(
                f"X와 y의 샘플 수가 일치하지 않습니다: {X.shape[0]} vs {len(y)}"
            )

        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # 랜덤 시드 설정
        rng = np.random.default_rng(self.random_state)

        # 1. 초기화: 균등 가중치
        sample_weights = np.ones(n_samples) / n_samples

        # 리스트 초기화
        self.estimators_ = []
        estimator_weights = []
        estimator_errors = []
        self.training_history_ = []

        for m in range(self.n_estimators):
            # 2a. 가중치 기반 부트스트랩 샘플링
            # (가중치를 직접 사용하는 대신 가중치 비례 샘플링)
            sample_indices = rng.choice(
                n_samples,
                size=n_samples,
                replace=True,
                p=sample_weights
            )
            X_sampled = X[sample_indices]
            y_sampled = y[sample_indices]

            # 2a. 학습기 학습
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.integers(0, 2**31) if self.random_state is not None else None
            )
            tree.fit(X_sampled, y_sampled)

            # 원본 데이터에 대한 예측
            y_pred = tree.predict(X)

            # 2b. 각 샘플의 손실 계산
            sample_losses = self._calculate_loss(y, y_pred)

            # 2c. 가중 평균 손실 계산
            avg_loss = np.sum(sample_weights * sample_losses)

            # 손실이 0.5 이상이면 학습 중단 (성능이 랜덤보다 나쁨)
            if avg_loss >= 0.5:
                if m == 0:
                    # 첫 번째 학습기도 실패하면 하나는 추가
                    self.estimators_.append(tree)
                    estimator_weights.append(1.0)
                    estimator_errors.append(avg_loss)
                break

            # 2d. 학습기 가중치(beta) 계산
            beta = avg_loss / (1 - avg_loss)
            # learning_rate 적용
            beta = beta ** self.learning_rate

            # 2e. 샘플 가중치 업데이트
            # 잘 예측된 샘플(낮은 손실)은 가중치 감소
            sample_weights = sample_weights * (beta ** (1 - sample_losses))

            # 가중치 정규화
            sample_weights = sample_weights / np.sum(sample_weights)

            # 학습기 저장
            self.estimators_.append(tree)
            estimator_weights.append(np.log(1 / beta) if beta > 0 else 10)  # log(1/β)
            estimator_errors.append(avg_loss)

            # 학습 과정 기록
            mse = np.mean((y - y_pred) ** 2)
            self.training_history_.append({
                'iteration': m + 1,
                'avg_loss': avg_loss,
                'beta': beta,
                'estimator_weight': estimator_weights[-1],
                'mse': mse,
                'rmse': np.sqrt(mse),
                'weight_entropy': -np.sum(sample_weights * np.log(sample_weights + 1e-10)),
                'max_weight': np.max(sample_weights),
                'min_weight': np.min(sample_weights)
            })

        # 배열로 변환
        self.estimator_weights_ = np.array(estimator_weights)
        self.estimator_errors_ = np.array(estimator_errors)

        # 피처 중요도 계산 (가중 평균)
        if len(self.estimators_) > 0:
            weighted_importances = np.zeros(n_features)
            total_weight = np.sum(self.estimator_weights_)

            for tree, weight in zip(self.estimators_, self.estimator_weights_):
                weighted_importances += weight * tree.feature_importances_

            self.feature_importances_ = weighted_importances / total_weight
        else:
            self.feature_importances_ = np.zeros(n_features)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        예측 수행 (가중 중앙값)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            예측할 데이터

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            예측값
        """
        if len(self.estimators_) == 0:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]

        # 각 학습기의 예측값 수집
        predictions = np.array([tree.predict(X) for tree in self.estimators_])

        # 가중 중앙값으로 최종 예측
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            y_pred[i] = self._weighted_median(
                predictions[:, i],
                self.estimator_weights_
            )

        return y_pred

    def staged_predict(self, X: np.ndarray) -> np.ndarray:
        """
        각 부스팅 라운드별 예측 반환 (시각화용)

        Returns
        -------
        predictions : ndarray of shape (n_estimators, n_samples)
            각 라운드까지의 누적 예측값들
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        n_estimators = len(self.estimators_)
        staged_predictions = np.zeros((n_estimators, n_samples))

        # 각 학습기의 예측값
        all_predictions = np.array([tree.predict(X) for tree in self.estimators_])

        # 누적 가중 중앙값 계산
        for m in range(n_estimators):
            current_predictions = all_predictions[:m + 1]
            current_weights = self.estimator_weights_[:m + 1]

            for i in range(n_samples):
                staged_predictions[m, i] = self._weighted_median(
                    current_predictions[:, i],
                    current_weights
                )

        return staged_predictions

    def __repr__(self) -> str:
        if len(self.estimators_) == 0:
            return "AdaBoostRegressor(not fitted)"

        return (
            f"AdaBoostRegressor("
            f"n_estimators={len(self.estimators_)}, "
            f"learning_rate={self.learning_rate}, "
            f"loss='{self.loss}')"
        )
