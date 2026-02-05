"""
Decision Tree Regressor - From Scratch Implementation
======================================================

CART (Classification and Regression Trees) 알고리즘 기반 결정 트리 구현.

수학적 배경:
-----------
분할 기준: MSE (Mean Squared Error) 감소 최대화

분할 전 MSE:
    MSE_parent = (1/n) * Σ(y_i - ȳ)²

분할 후 가중 MSE:
    MSE_split = (n_left/n) * MSE_left + (n_right/n) * MSE_right

정보 이득 (Information Gain):
    Gain = MSE_parent - MSE_split

최적 분할: Gain이 최대인 (feature, threshold) 선택

예측:
    leaf_prediction = mean(y_samples in leaf)

Author: ML From Scratch Project
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass


@dataclass
class TreeNode:
    """결정 트리의 노드를 표현하는 클래스"""

    # 분할 정보 (내부 노드용)
    feature_idx: Optional[int] = None    # 분할에 사용된 피처 인덱스
    threshold: Optional[float] = None    # 분할 임계값

    # 자식 노드
    left: Optional['TreeNode'] = None    # 왼쪽 자식 (값 <= threshold)
    right: Optional['TreeNode'] = None   # 오른쪽 자식 (값 > threshold)

    # 리프 노드 정보
    value: Optional[float] = None        # 리프 노드의 예측값
    n_samples: int = 0                   # 노드에 도달한 샘플 수
    mse: float = 0.0                     # 노드의 MSE
    depth: int = 0                       # 노드의 깊이

    def is_leaf(self) -> bool:
        """리프 노드인지 확인"""
        return self.left is None and self.right is None


class DecisionTreeRegressor:
    """
    CART 기반 결정 트리 회귀 모델 (From Scratch)

    Parameters
    ----------
    max_depth : int, default=None
        트리의 최대 깊이. None이면 제한 없음.

    min_samples_split : int, default=2
        내부 노드를 분할하기 위한 최소 샘플 수.

    min_samples_leaf : int, default=1
        리프 노드에 있어야 하는 최소 샘플 수.

    min_impurity_decrease : float, default=0.0
        분할을 수행하기 위한 최소 불순도 감소량.

    max_features : int or float or str, default=None
        각 분할에서 고려할 피처 수.
        - None: 모든 피처 사용
        - int: 해당 수의 피처 사용
        - float: 비율로 피처 수 결정
        - 'sqrt': sqrt(n_features)
        - 'log2': log2(n_features)

    random_state : int, default=None
        랜덤 시드 (피처 서브샘플링용)

    Attributes
    ----------
    root_ : TreeNode
        학습된 트리의 루트 노드

    n_features_ : int
        학습에 사용된 피처 수

    feature_importances_ : ndarray of shape (n_features,)
        피처 중요도 (불순도 감소 기반)

    tree_stats_ : dict
        트리 통계 (깊이, 노드 수, 리프 수 등)

    Examples
    --------
    >>> from ml_from_scratch import DecisionTreeRegressor
    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> y = np.array([1.1, 2.0, 3.1, 3.9, 5.0])
    >>> tree = DecisionTreeRegressor(max_depth=2)
    >>> tree.fit(X, y)
    >>> tree.predict(np.array([[2.5]]))
    array([2.55])
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[Any] = None,
        random_state: Optional[int] = None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state

        # 학습 후 설정되는 속성들
        self.root_: Optional[TreeNode] = None
        self.n_features_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None
        self.tree_stats_: Dict = {}
        self._rng: Optional[np.random.Generator] = None

        # 학습 과정 기록 (시각화용)
        self.training_history_: List[Dict] = []

    def _calculate_mse(self, y: np.ndarray) -> float:
        """
        MSE (Mean Squared Error) 계산

        MSE = (1/n) * Σ(y_i - ȳ)²

        이것은 분산(variance)과 동일함
        """
        if len(y) == 0:
            return 0.0
        return np.var(y)

    def _calculate_split_mse(
        self,
        y_left: np.ndarray,
        y_right: np.ndarray
    ) -> float:
        """
        분할 후 가중 MSE 계산

        MSE_split = (n_left/n) * MSE_left + (n_right/n) * MSE_right
        """
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        if n_total == 0:
            return 0.0

        mse_left = self._calculate_mse(y_left)
        mse_right = self._calculate_mse(y_right)

        return (n_left / n_total) * mse_left + (n_right / n_total) * mse_right

    def _get_n_features_to_sample(self, n_features: int) -> int:
        """각 분할에서 고려할 피처 수 결정"""
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        else:
            return n_features

    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        최적의 분할점 탐색

        모든 피처와 가능한 임계값에 대해:
        1. 데이터를 left(<=threshold), right(>threshold)로 분할
        2. 정보 이득 계산
        3. 최대 정보 이득을 주는 (feature, threshold) 반환

        Returns
        -------
        best_feature : int or None
            최적 분할 피처 인덱스
        best_threshold : float or None
            최적 분할 임계값
        best_gain : float
            최대 정보 이득
        """
        n_samples, n_features = X.shape

        # 분할 전 MSE
        mse_parent = self._calculate_mse(y)

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        # 피처 서브샘플링
        n_features_to_sample = self._get_n_features_to_sample(n_features)
        if n_features_to_sample < n_features and self._rng is not None:
            feature_indices = self._rng.choice(
                n_features, n_features_to_sample, replace=False
            )
        else:
            feature_indices = np.arange(n_features)

        for feature_idx in feature_indices:
            # 해당 피처의 고유값들을 임계값 후보로 사용
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            # 인접한 고유값들의 중간점을 임계값으로 사용
            if len(unique_values) > 1:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            else:
                continue

            for threshold in thresholds:
                # 분할 수행
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                y_left = y[left_mask]
                y_right = y[right_mask]

                # min_samples_leaf 조건 확인
                if len(y_left) < self.min_samples_leaf or \
                   len(y_right) < self.min_samples_leaf:
                    continue

                # 정보 이득 계산
                mse_split = self._calculate_split_mse(y_left, y_right)
                gain = mse_parent - mse_split

                # 최적 분할 갱신
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int = 0
    ) -> TreeNode:
        """
        재귀적으로 결정 트리 구축

        종료 조건:
        1. max_depth 도달
        2. 샘플 수 < min_samples_split
        3. 모든 타겟값이 동일
        4. 정보 이득 < min_impurity_decrease
        """
        n_samples = len(y)
        mse = self._calculate_mse(y)

        # 노드 생성
        node = TreeNode(
            value=np.mean(y),
            n_samples=n_samples,
            mse=mse,
            depth=depth
        )

        # 종료 조건 확인
        should_stop = (
            (self.max_depth is not None and depth >= self.max_depth) or
            n_samples < self.min_samples_split or
            mse == 0  # 모든 값이 동일
        )

        if should_stop:
            # 학습 과정 기록
            self.training_history_.append({
                'depth': depth,
                'n_samples': n_samples,
                'mse': mse,
                'action': 'leaf',
                'value': node.value
            })
            return node

        # 최적 분할 탐색
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)

        # 유효한 분할이 없거나 정보 이득이 부족한 경우
        if best_feature is None or best_gain < self.min_impurity_decrease:
            self.training_history_.append({
                'depth': depth,
                'n_samples': n_samples,
                'mse': mse,
                'action': 'leaf',
                'value': node.value
            })
            return node

        # 분할 수행
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # 학습 과정 기록
        self.training_history_.append({
            'depth': depth,
            'n_samples': n_samples,
            'mse': mse,
            'action': 'split',
            'feature': best_feature,
            'threshold': best_threshold,
            'gain': best_gain,
            'n_left': np.sum(left_mask),
            'n_right': np.sum(right_mask)
        })

        # 분할 정보 저장
        node.feature_idx = best_feature
        node.threshold = best_threshold

        # 재귀적으로 자식 노드 구축
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _calculate_feature_importances(self, node: TreeNode) -> np.ndarray:
        """
        피처 중요도 계산 (불순도 감소 기반)

        importance[i] = Σ (n_samples * mse_decrease) for splits using feature i
        """
        importances = np.zeros(self.n_features_)

        def _traverse(node: TreeNode):
            if node.is_leaf():
                return

            # 이 분할로 인한 불순도 감소량
            mse_decrease = node.mse - (
                (node.left.n_samples / node.n_samples) * node.left.mse +
                (node.right.n_samples / node.n_samples) * node.right.mse
            )

            # 가중치: 노드에 도달한 샘플 비율
            importances[node.feature_idx] += node.n_samples * mse_decrease

            # 재귀적으로 자식 노드 탐색
            _traverse(node.left)
            _traverse(node.right)

        _traverse(node)

        # 정규화
        total = np.sum(importances)
        if total > 0:
            importances /= total

        return importances

    def _calculate_tree_stats(self, node: TreeNode) -> Dict:
        """트리 통계 계산"""
        stats = {
            'max_depth': 0,
            'n_nodes': 0,
            'n_leaves': 0,
            'n_internal': 0,
            'avg_leaf_depth': 0,
            'leaf_depths': []
        }

        def _traverse(node: TreeNode, depth: int):
            stats['n_nodes'] += 1
            stats['max_depth'] = max(stats['max_depth'], depth)

            if node.is_leaf():
                stats['n_leaves'] += 1
                stats['leaf_depths'].append(depth)
            else:
                stats['n_internal'] += 1
                _traverse(node.left, depth + 1)
                _traverse(node.right, depth + 1)

        _traverse(node, 0)

        if stats['n_leaves'] > 0:
            stats['avg_leaf_depth'] = np.mean(stats['leaf_depths'])

        return stats

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressor':
        """
        결정 트리 학습

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            학습 데이터
        y : ndarray of shape (n_samples,)
            타겟 값

        Returns
        -------
        self : DecisionTreeRegressor
            학습된 모델
        """
        # 입력 검증
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if X.shape[0] != len(y):
            raise ValueError(
                f"X와 y의 샘플 수가 일치하지 않습니다: {X.shape[0]} vs {len(y)}"
            )

        self.n_features_ = X.shape[1]

        # 랜덤 시드 설정
        if self.random_state is not None:
            self._rng = np.random.default_rng(self.random_state)

        # 학습 기록 초기화
        self.training_history_ = []

        # 트리 구축
        self.root_ = self._build_tree(X, y)

        # 피처 중요도 계산
        self.feature_importances_ = self._calculate_feature_importances(self.root_)

        # 트리 통계 계산
        self.tree_stats_ = self._calculate_tree_stats(self.root_)

        return self

    def _predict_single(self, x: np.ndarray) -> float:
        """단일 샘플 예측"""
        node = self.root_

        while not node.is_leaf():
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.value

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
        if self.root_ is None:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        return np.array([self._predict_single(x) for x in X])

    def get_depth(self) -> int:
        """트리의 최대 깊이 반환"""
        if self.root_ is None:
            return 0
        return self.tree_stats_.get('max_depth', 0)

    def get_n_leaves(self) -> int:
        """리프 노드 수 반환"""
        if self.root_ is None:
            return 0
        return self.tree_stats_.get('n_leaves', 0)

    def export_tree_structure(self) -> Dict:
        """
        트리 구조를 딕셔너리로 내보내기 (시각화용)
        """
        def _node_to_dict(node: TreeNode) -> Dict:
            result = {
                'value': node.value,
                'n_samples': node.n_samples,
                'mse': node.mse,
                'depth': node.depth,
                'is_leaf': node.is_leaf()
            }

            if not node.is_leaf():
                result['feature_idx'] = node.feature_idx
                result['threshold'] = node.threshold
                result['left'] = _node_to_dict(node.left)
                result['right'] = _node_to_dict(node.right)

            return result

        if self.root_ is None:
            return {}

        return _node_to_dict(self.root_)

    def __repr__(self) -> str:
        if self.root_ is None:
            return "DecisionTreeRegressor(not fitted)"

        return (
            f"DecisionTreeRegressor("
            f"depth={self.get_depth()}, "
            f"n_leaves={self.get_n_leaves()}, "
            f"n_features={self.n_features_})"
        )
