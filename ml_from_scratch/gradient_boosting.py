"""
Gradient Boosting Regressor - From Scratch Implementation
=========================================================

Gradient Boosting은 잔차(residual)를 순차적으로 학습하는 앙상블 방법입니다.

수학적 배경:
-----------
손실 함수 (MSE):
    L(y, F) = (1/2) * (y - F(x))²

음의 그래디언트 (= 잔차):
    r = -∂L/∂F = y - F(x)

업데이트 규칙:
    F_m(x) = F_{m-1}(x) + η * h_m(x)

여기서:
    - F_m(x): m번째 반복 후의 예측
    - η: 학습률 (learning rate)
    - h_m(x): 잔차를 예측하도록 학습된 m번째 트리

알고리즘:
--------
1. 초기화: F_0(x) = mean(y)
2. for m = 1 to M:
   a. 잔차 계산: r_i = y_i - F_{m-1}(x_i)
   b. 잔차에 대해 트리 h_m 학습
   c. 예측 업데이트: F_m(x) = F_{m-1}(x) + η * h_m(x)
3. 최종 예측: F_M(x)

Author: ML From Scratch Project
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from .decision_tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    """
    Gradient Boosting 회귀 모델 (From Scratch)

    Parameters
    ----------
    n_estimators : int, default=100
        부스팅 라운드 수 (트리 개수)

    learning_rate : float, default=0.1
        각 트리의 기여도를 조절하는 축소 계수 (shrinkage)
        작은 값일수록 더 많은 트리가 필요하지만 일반화 성능이 좋아질 수 있음

    max_depth : int, default=3
        각 트리의 최대 깊이
        Gradient Boosting에서는 보통 얕은 트리(stump 또는 depth 3-5)를 사용

    min_samples_split : int, default=2
        내부 노드를 분할하기 위한 최소 샘플 수

    min_samples_leaf : int, default=1
        리프 노드에 있어야 하는 최소 샘플 수

    subsample : float, default=1.0
        각 트리 학습에 사용할 샘플의 비율 (Stochastic GB)
        1.0 미만이면 확률적 그래디언트 부스팅이 됨

    max_features : int or float or str, default=None
        각 분할에서 고려할 피처 수

    random_state : int, default=None
        랜덤 시드

    verbose : int, default=0
        학습 과정 출력 수준 (0: 없음, 1: 진행률)

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        학습된 트리들

    train_scores_ : list of float
        각 라운드 후의 학습 MSE

    feature_importances_ : ndarray of shape (n_features,)
        피처 중요도 (모든 트리의 평균)

    init_prediction_ : float
        초기 예측값 (타겟의 평균)

    Examples
    --------
    >>> from ml_from_scratch import GradientBoostingRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = X[:, 0] * 2 + X[:, 1] + np.random.randn(100) * 0.1
    >>> gb = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1)
    >>> gb.fit(X, y)
    >>> predictions = gb.predict(X[:5])
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        max_features: Optional[str] = None,
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state
        self.verbose = verbose

        # 학습 후 설정되는 속성들
        self.estimators_: List[DecisionTreeRegressor] = []
        self.train_scores_: List[float] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.init_prediction_: float = 0.0
        self.n_features_: int = 0

        # 학습 과정 기록 (시각화용)
        self.training_history_: List[Dict] = []

    def _calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MSE 계산"""
        return np.mean((y_true - y_pred) ** 2)

    def _calculate_negative_gradient(
        self,
        y: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        음의 그래디언트 계산 (MSE 손실의 경우 = 잔차)

        L = (1/2) * (y - F)²
        ∂L/∂F = -(y - F) = F - y
        -∂L/∂F = y - F = residual

        Returns
        -------
        residuals : ndarray
            현재 예측에 대한 잔차
        """
        return y - y_pred

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'GradientBoostingRegressor':
        """
        Gradient Boosting 모델 학습

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            학습 데이터
        y : ndarray of shape (n_samples,)
            타겟 값
        X_val : ndarray, optional
            검증 데이터 (조기 종료 또는 모니터링용)
        y_val : ndarray, optional
            검증 타겟

        Returns
        -------
        self : GradientBoostingRegressor
            학습된 모델
        """
        # 입력 검증 및 변환
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

        # 초기화: F_0(x) = mean(y)
        self.init_prediction_ = np.mean(y)
        y_pred = np.full(n_samples, self.init_prediction_)

        # 검증 데이터 초기화
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val).ravel()
            y_val_pred = np.full(len(y_val), self.init_prediction_)
            val_scores = []

        # 리스트 초기화
        self.estimators_ = []
        self.train_scores_ = []
        self.training_history_ = []

        # 초기 점수 기록
        init_mse = self._calculate_mse(y, y_pred)
        self.train_scores_.append(init_mse)

        if self.verbose > 0:
            print(f"초기 MSE: {init_mse:.4f}")

        # 부스팅 라운드
        for m in range(self.n_estimators):
            # 1. 음의 그래디언트(잔차) 계산
            residuals = self._calculate_negative_gradient(y, y_pred)

            # 2. 서브샘플링 (Stochastic GB)
            if self.subsample < 1.0:
                n_subsample = int(n_samples * self.subsample)
                sample_indices = rng.choice(n_samples, n_subsample, replace=False)
                X_subsample = X[sample_indices]
                residuals_subsample = residuals[sample_indices]
            else:
                X_subsample = X
                residuals_subsample = residuals

            # 3. 잔차에 대해 트리 학습
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=rng.integers(0, 2**31) if self.random_state is not None else None
            )
            tree.fit(X_subsample, residuals_subsample)

            # 4. 예측 업데이트: F_m(x) = F_{m-1}(x) + η * h_m(x)
            tree_prediction = tree.predict(X)
            y_pred = y_pred + self.learning_rate * tree_prediction

            # 트리 저장
            self.estimators_.append(tree)

            # 학습 점수 기록
            train_mse = self._calculate_mse(y, y_pred)
            self.train_scores_.append(train_mse)

            # 검증 점수 계산 (옵션)
            val_mse = None
            if X_val is not None and y_val is not None:
                y_val_pred = y_val_pred + self.learning_rate * tree.predict(X_val)
                val_mse = self._calculate_mse(y_val, y_val_pred)
                val_scores.append(val_mse)

            # 학습 과정 기록
            history_entry = {
                'iteration': m + 1,
                'train_mse': train_mse,
                'train_rmse': np.sqrt(train_mse),
                'residual_mean': np.mean(residuals),
                'residual_std': np.std(residuals),
                'tree_depth': tree.get_depth(),
                'tree_n_leaves': tree.get_n_leaves()
            }
            if val_mse is not None:
                history_entry['val_mse'] = val_mse
                history_entry['val_rmse'] = np.sqrt(val_mse)

            self.training_history_.append(history_entry)

            # 진행 상황 출력
            if self.verbose > 0 and (m + 1) % max(1, self.n_estimators // 10) == 0:
                msg = f"라운드 {m + 1}/{self.n_estimators}, Train MSE: {train_mse:.4f}"
                if val_mse is not None:
                    msg += f", Val MSE: {val_mse:.4f}"
                print(msg)

        # 피처 중요도 계산 (모든 트리의 평균)
        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.estimators_],
            axis=0
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        예측 수행

        F_M(x) = F_0(x) + η * Σ h_m(x)

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

        # 초기 예측
        y_pred = np.full(X.shape[0], self.init_prediction_)

        # 각 트리의 예측을 누적
        for tree in self.estimators_:
            y_pred = y_pred + self.learning_rate * tree.predict(X)

        return y_pred

    def staged_predict(self, X: np.ndarray) -> np.ndarray:
        """
        각 부스팅 라운드별 예측 반환 (시각화용)

        Returns
        -------
        predictions : ndarray of shape (n_estimators + 1, n_samples)
            각 라운드 후의 예측값들
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        predictions = np.zeros((len(self.estimators_) + 1, n_samples))

        # 초기 예측
        predictions[0] = self.init_prediction_

        # 각 트리 추가 후의 예측
        y_pred = np.full(n_samples, self.init_prediction_)
        for i, tree in enumerate(self.estimators_):
            y_pred = y_pred + self.learning_rate * tree.predict(X)
            predictions[i + 1] = y_pred

        return predictions

    def get_training_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        학습 곡선 데이터 반환

        Returns
        -------
        iterations : ndarray
            반복 횟수
        train_mse : ndarray
            각 반복에서의 학습 MSE
        """
        iterations = np.arange(len(self.train_scores_))
        return iterations, np.array(self.train_scores_)

    def __repr__(self) -> str:
        if len(self.estimators_) == 0:
            return "GradientBoostingRegressor(not fitted)"

        return (
            f"GradientBoostingRegressor("
            f"n_estimators={len(self.estimators_)}, "
            f"learning_rate={self.learning_rate}, "
            f"max_depth={self.max_depth})"
        )
