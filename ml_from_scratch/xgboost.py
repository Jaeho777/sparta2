"""
XGBoost-style Regressor - From Scratch Implementation
=====================================================

XGBoost의 핵심 아이디어를 구현: 2차 테일러 전개 + 정규화

수학적 배경:
-----------
1. 목적 함수:
   Obj = Σ L(y_i, ŷ_i) + Σ Ω(f_k)

   손실 함수:
   L(y, ŷ) = (1/2) * (y - ŷ)²

   정규화 항:
   Ω(f) = γT + (1/2)λ||w||²
   여기서 T = 리프 수, w = 리프 가중치

2. 2차 테일러 전개:
   L(y, ŷ + Δ) ≈ L(y, ŷ) + g·Δ + (1/2)h·Δ²

   MSE의 경우:
   g = ∂L/∂ŷ = -(y - ŷ) = ŷ - y (gradient)
   h = ∂²L/∂ŷ² = 1 (hessian)

3. 최적 리프 가중치:
   w* = -G / (H + λ)
   여기서 G = Σg_i, H = Σh_i (리프에 속한 샘플들의 합)

4. 분할 이득 (Gain):
   Gain = (1/2) * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ

Author: ML From Scratch Project
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class XGBTreeNode:
    """XGBoost 트리 노드"""
    # 분할 정보
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None

    # 자식 노드
    left: Optional['XGBTreeNode'] = None
    right: Optional['XGBTreeNode'] = None

    # 리프 정보
    weight: float = 0.0  # 최적 리프 가중치 w*
    G: float = 0.0       # 그래디언트 합
    H: float = 0.0       # 헤시안 합
    n_samples: int = 0
    depth: int = 0
    gain: float = 0.0    # 분할 이득

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class XGBTreeRegressor:
    """
    XGBoost 스타일의 단일 트리 (정규화 적용)
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 2,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        reg_gamma: float = 0.0,
        colsample_bynode: float = 1.0,
        random_state: Optional[int] = None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_child_weight = min_child_weight  # 리프의 최소 H 합
        self.reg_lambda = reg_lambda  # L2 정규화
        self.reg_gamma = reg_gamma    # 분할 복잡도 페널티
        self.colsample_bynode = colsample_bynode
        self.random_state = random_state

        self.root_: Optional[XGBTreeNode] = None
        self.n_features_: int = 0
        self._rng: Optional[np.random.Generator] = None

    def _calculate_leaf_weight(self, G: float, H: float) -> float:
        """
        최적 리프 가중치 계산

        w* = -G / (H + λ)
        """
        return -G / (H + self.reg_lambda)

    def _calculate_gain(
        self,
        G: float, H: float,
        G_left: float, H_left: float,
        G_right: float, H_right: float
    ) -> float:
        """
        분할 이득 계산

        Gain = (1/2) * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ
        """
        def score(g, h):
            return (g ** 2) / (h + self.reg_lambda)

        gain = 0.5 * (
            score(G_left, H_left) +
            score(G_right, H_right) -
            score(G, H)
        ) - self.reg_gamma

        return gain

    def _find_best_split(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        최적 분할점 탐색 (2차 근사 기반)
        """
        n_samples, n_features = X.shape

        G = np.sum(gradients)
        H = np.sum(hessians)

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        # 피처 서브샘플링
        if self.colsample_bynode < 1.0 and self._rng is not None:
            n_features_to_use = max(1, int(n_features * self.colsample_bynode))
            feature_indices = self._rng.choice(n_features, n_features_to_use, replace=False)
        else:
            feature_indices = np.arange(n_features)

        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]

            # 정렬된 인덱스
            sorted_indices = np.argsort(feature_values)
            sorted_values = feature_values[sorted_indices]
            sorted_gradients = gradients[sorted_indices]
            sorted_hessians = hessians[sorted_indices]

            # 누적 합 (왼쪽 → 오른쪽)
            G_left = 0.0
            H_left = 0.0

            for i in range(n_samples - 1):
                G_left += sorted_gradients[i]
                H_left += sorted_hessians[i]
                G_right = G - G_left
                H_right = H - H_left

                # 같은 값이면 건너뛰기
                if sorted_values[i] == sorted_values[i + 1]:
                    continue

                # min_child_weight 조건
                if H_left < self.min_child_weight or H_right < self.min_child_weight:
                    continue

                # 분할 이득 계산
                gain = self._calculate_gain(G, H, G_left, H_left, G_right, H_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    # 중간값을 임계값으로
                    best_threshold = (sorted_values[i] + sorted_values[i + 1]) / 2

        return best_feature, best_threshold, best_gain

    def _build_tree(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        depth: int = 0
    ) -> XGBTreeNode:
        """재귀적 트리 구축"""
        n_samples = len(gradients)
        G = np.sum(gradients)
        H = np.sum(hessians)

        # 노드 생성
        node = XGBTreeNode(
            weight=self._calculate_leaf_weight(G, H),
            G=G,
            H=H,
            n_samples=n_samples,
            depth=depth
        )

        # 종료 조건
        should_stop = (
            depth >= self.max_depth or
            n_samples < self.min_samples_split or
            H < self.min_child_weight
        )

        if should_stop:
            return node

        # 최적 분할 탐색
        best_feature, best_threshold, best_gain = self._find_best_split(
            X, gradients, hessians
        )

        # 유효한 분할이 없거나 이득이 없으면 리프
        if best_feature is None or best_gain <= 0:
            return node

        # 분할 수행
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # 분할 정보 저장
        node.feature_idx = best_feature
        node.threshold = best_threshold
        node.gain = best_gain

        # 재귀적으로 자식 구축
        node.left = self._build_tree(
            X[left_mask], gradients[left_mask], hessians[left_mask], depth + 1
        )
        node.right = self._build_tree(
            X[right_mask], gradients[right_mask], hessians[right_mask], depth + 1
        )

        return node

    def fit(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray
    ) -> 'XGBTreeRegressor':
        """그래디언트와 헤시안을 받아 트리 학습"""
        X = np.asarray(X)
        gradients = np.asarray(gradients).ravel()
        hessians = np.asarray(hessians).ravel()

        self.n_features_ = X.shape[1]

        if self.random_state is not None:
            self._rng = np.random.default_rng(self.random_state)

        self.root_ = self._build_tree(X, gradients, hessians)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 (리프 가중치 반환)"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        def _predict_single(x: np.ndarray) -> float:
            node = self.root_
            while not node.is_leaf():
                if x[node.feature_idx] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            return node.weight

        return np.array([_predict_single(x) for x in X])


class XGBoostRegressor:
    """
    XGBoost 회귀 모델 (From Scratch)

    2차 테일러 전개와 정규화를 사용하는 그래디언트 부스팅

    Parameters
    ----------
    n_estimators : int, default=100
        부스팅 라운드 수

    learning_rate : float, default=0.3
        축소 계수 (eta)

    max_depth : int, default=6
        각 트리의 최대 깊이

    min_child_weight : float, default=1.0
        리프의 최소 헤시안 합 (과적합 방지)

    reg_lambda : float, default=1.0
        L2 정규화 계수 (리프 가중치)

    reg_gamma : float, default=0.0
        분할 복잡도 페널티 (pruning)

    subsample : float, default=1.0
        행 서브샘플링 비율

    colsample_bytree : float, default=1.0
        트리별 컬럼 서브샘플링 비율

    colsample_bynode : float, default=1.0
        노드별 컬럼 서브샘플링 비율

    random_state : int, default=None
        랜덤 시드

    verbose : int, default=0
        출력 수준

    Attributes
    ----------
    estimators_ : list of XGBTreeRegressor
        학습된 트리들

    train_scores_ : list of float
        각 라운드 후의 학습 MSE

    feature_importances_ : ndarray
        피처 중요도 (gain 기반)

    Examples
    --------
    >>> from ml_from_scratch import XGBoostRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = X[:, 0] * 2 + X[:, 1] + np.random.randn(100) * 0.1
    >>> xgb = XGBoostRegressor(n_estimators=50, learning_rate=0.1)
    >>> xgb.fit(X, y)
    >>> predictions = xgb.predict(X[:5])
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        reg_gamma: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        colsample_bynode: float = 1.0,
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bynode = colsample_bynode
        self.random_state = random_state
        self.verbose = verbose

        # 학습 후 설정되는 속성들
        self.estimators_: List[XGBTreeRegressor] = []
        self.train_scores_: List[float] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.base_prediction_: float = 0.0
        self.n_features_: int = 0

        # 학습 과정 기록
        self.training_history_: List[Dict] = []

    def _calculate_gradients(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        MSE 손실에 대한 그래디언트와 헤시안 계산

        L = (1/2) * (y - ŷ)²
        g = ∂L/∂ŷ = ŷ - y
        h = ∂²L/∂ŷ² = 1
        """
        gradients = y_pred - y_true
        hessians = np.ones_like(y_true)
        return gradients, hessians

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'XGBoostRegressor':
        """
        XGBoost 모델 학습

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            학습 데이터
        y : ndarray of shape (n_samples,)
            타겟 값
        X_val : ndarray, optional
            검증 데이터
        y_val : ndarray, optional
            검증 타겟

        Returns
        -------
        self : XGBoostRegressor
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

        # 랜덤 시드
        rng = np.random.default_rng(self.random_state)

        # 초기 예측: 타겟의 평균
        self.base_prediction_ = np.mean(y)
        y_pred = np.full(n_samples, self.base_prediction_)

        # 검증 데이터 초기화
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val).ravel()
            y_val_pred = np.full(len(y_val), self.base_prediction_)

        # 리스트 초기화
        self.estimators_ = []
        self.train_scores_ = []
        self.training_history_ = []
        feature_gains = np.zeros(n_features)

        # 초기 MSE
        init_mse = np.mean((y - y_pred) ** 2)
        self.train_scores_.append(init_mse)

        if self.verbose > 0:
            print(f"초기 MSE: {init_mse:.4f}")

        # 부스팅 라운드
        for m in range(self.n_estimators):
            # 1. 그래디언트와 헤시안 계산
            gradients, hessians = self._calculate_gradients(y, y_pred)

            # 2. 행 서브샘플링
            if self.subsample < 1.0:
                n_subsample = int(n_samples * self.subsample)
                sample_indices = rng.choice(n_samples, n_subsample, replace=False)
                X_sub = X[sample_indices]
                g_sub = gradients[sample_indices]
                h_sub = hessians[sample_indices]
            else:
                X_sub = X
                g_sub = gradients
                h_sub = hessians

            # 3. 컬럼 서브샘플링 (트리별)
            if self.colsample_bytree < 1.0:
                n_cols = max(1, int(n_features * self.colsample_bytree))
                col_indices = rng.choice(n_features, n_cols, replace=False)
                X_sub = X_sub[:, col_indices]
                X_pred = X[:, col_indices]
            else:
                col_indices = np.arange(n_features)
                X_pred = X

            # 4. 트리 학습
            tree = XGBTreeRegressor(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                reg_gamma=self.reg_gamma,
                colsample_bynode=self.colsample_bynode,
                random_state=rng.integers(0, 2**31) if self.random_state is not None else None
            )
            tree.fit(X_sub, g_sub, h_sub)

            # 5. 예측 업데이트
            tree_pred = tree.predict(X_pred)
            y_pred = y_pred + self.learning_rate * tree_pred

            # 트리 저장
            self.estimators_.append((tree, col_indices))

            # 피처 중요도 누적 (gain 기반)
            self._accumulate_feature_gains(tree.root_, col_indices, feature_gains)

            # 학습 점수 기록
            train_mse = np.mean((y - y_pred) ** 2)
            self.train_scores_.append(train_mse)

            # 검증 점수
            val_mse = None
            if X_val is not None and y_val is not None:
                X_val_sub = X_val[:, col_indices]
                y_val_pred = y_val_pred + self.learning_rate * tree.predict(X_val_sub)
                val_mse = np.mean((y_val - y_val_pred) ** 2)

            # 기록
            history_entry = {
                'iteration': m + 1,
                'train_mse': train_mse,
                'train_rmse': np.sqrt(train_mse),
                'gradient_mean': np.mean(gradients),
                'gradient_std': np.std(gradients)
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

        # 피처 중요도 정규화
        total_gain = np.sum(feature_gains)
        if total_gain > 0:
            self.feature_importances_ = feature_gains / total_gain
        else:
            self.feature_importances_ = np.zeros(n_features)

        return self

    def _accumulate_feature_gains(
        self,
        node: XGBTreeNode,
        col_indices: np.ndarray,
        feature_gains: np.ndarray
    ):
        """트리의 각 분할에서 피처별 gain 누적"""
        if node.is_leaf():
            return

        # 원래 피처 인덱스로 매핑
        original_feature_idx = col_indices[node.feature_idx]
        feature_gains[original_feature_idx] += node.gain

        self._accumulate_feature_gains(node.left, col_indices, feature_gains)
        self._accumulate_feature_gains(node.right, col_indices, feature_gains)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        예측 수행

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
        y_pred = np.full(X.shape[0], self.base_prediction_)

        # 각 트리의 예측 누적
        for tree, col_indices in self.estimators_:
            X_sub = X[:, col_indices]
            y_pred = y_pred + self.learning_rate * tree.predict(X_sub)

        return y_pred

    def staged_predict(self, X: np.ndarray) -> np.ndarray:
        """각 부스팅 라운드별 예측 반환"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        predictions = np.zeros((len(self.estimators_) + 1, n_samples))

        # 초기 예측
        predictions[0] = self.base_prediction_
        y_pred = np.full(n_samples, self.base_prediction_)

        for i, (tree, col_indices) in enumerate(self.estimators_):
            X_sub = X[:, col_indices]
            y_pred = y_pred + self.learning_rate * tree.predict(X_sub)
            predictions[i + 1] = y_pred

        return predictions

    def get_training_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """학습 곡선 데이터 반환"""
        iterations = np.arange(len(self.train_scores_))
        return iterations, np.array(self.train_scores_)

    def __repr__(self) -> str:
        if len(self.estimators_) == 0:
            return "XGBoostRegressor(not fitted)"

        return (
            f"XGBoostRegressor("
            f"n_estimators={len(self.estimators_)}, "
            f"learning_rate={self.learning_rate}, "
            f"max_depth={self.max_depth}, "
            f"reg_lambda={self.reg_lambda})"
        )
