"""
Random Forest Regressor - From Scratch Implementation
=====================================================

배깅(Bootstrap Aggregating) + 랜덤 피처 선택을 결합한 앙상블 방법

수학적 배경:
-----------
1. 배깅 (Bootstrap Aggregating):
   - 원본 데이터에서 복원 추출로 n개의 부트스트랩 샘플 생성
   - 각 샘플로 독립적인 트리 학습
   - 분산 감소: Var(평균) = Var(개별) / n (독립인 경우)

2. 랜덤 피처 선택:
   - 각 분할에서 sqrt(n_features) 또는 일부 피처만 고려
   - 트리 간 상관관계 감소 → 앙상블 효과 증대

3. Out-of-Bag (OOB) 오차:
   - 각 트리 학습에 사용되지 않은 샘플(~37%)로 오차 추정
   - 별도의 검증 세트 없이 일반화 오차 추정 가능

   P(샘플이 선택되지 않음) = (1 - 1/n)^n ≈ e^{-1} ≈ 0.368

4. 최종 예측:
   ŷ = (1/M) * Σ h_m(x)
   (모든 트리 예측의 평균)

Author: ML From Scratch Project
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from .decision_tree import DecisionTreeRegressor


class RandomForestRegressor:
    """
    Random Forest 회귀 모델 (From Scratch)

    Parameters
    ----------
    n_estimators : int, default=100
        트리 개수

    max_depth : int, default=None
        각 트리의 최대 깊이. None이면 완전히 확장

    min_samples_split : int, default=2
        내부 노드를 분할하기 위한 최소 샘플 수

    min_samples_leaf : int, default=1
        리프 노드에 있어야 하는 최소 샘플 수

    max_features : str or int or float, default='sqrt'
        각 분할에서 고려할 피처 수
        - 'sqrt': sqrt(n_features)
        - 'log2': log2(n_features)
        - int: 해당 수
        - float: 비율

    bootstrap : bool, default=True
        부트스트랩 샘플 사용 여부

    oob_score : bool, default=False
        Out-of-Bag 점수 계산 여부

    random_state : int, default=None
        랜덤 시드

    n_jobs : int, default=1
        병렬 처리 수 (현재 구현에서는 미사용)

    verbose : int, default=0
        출력 수준

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        학습된 트리들

    feature_importances_ : ndarray of shape (n_features,)
        피처 중요도 (모든 트리의 평균)

    oob_score_ : float
        Out-of-Bag R² 점수 (oob_score=True인 경우)

    oob_prediction_ : ndarray
        각 샘플의 OOB 예측값

    Examples
    --------
    >>> from ml_from_scratch import RandomForestRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = X[:, 0] * 2 + X[:, 1] + np.random.randn(100) * 0.1
    >>> rf = RandomForestRegressor(n_estimators=50)
    >>> rf.fit(X, y)
    >>> predictions = rf.predict(X[:5])
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        verbose: int = 0
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        # 학습 후 설정되는 속성들
        self.estimators_: List[DecisionTreeRegressor] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.oob_score_: Optional[float] = None
        self.oob_prediction_: Optional[np.ndarray] = None
        self.n_features_: int = 0

        # 학습 과정 기록
        self.training_history_: List[Dict] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestRegressor':
        """
        Random Forest 모델 학습

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            학습 데이터
        y : ndarray of shape (n_samples,)
            타겟 값

        Returns
        -------
        self : RandomForestRegressor
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

        # OOB 예측을 위한 배열
        if self.oob_score:
            oob_predictions_sum = np.zeros(n_samples)
            oob_predictions_count = np.zeros(n_samples)

        # 리스트 초기화
        self.estimators_ = []
        self.training_history_ = []
        all_importances = []

        if self.verbose > 0:
            print(f"Random Forest 학습 시작: {self.n_estimators}개 트리")

        # 각 트리 학습
        for m in range(self.n_estimators):
            # 부트스트랩 샘플링
            if self.bootstrap:
                sample_indices = rng.choice(n_samples, n_samples, replace=True)
                X_sample = X[sample_indices]
                y_sample = y[sample_indices]

                # OOB 인덱스 (선택되지 않은 샘플)
                if self.oob_score:
                    oob_indices = np.setdiff1d(np.arange(n_samples), np.unique(sample_indices))
            else:
                X_sample = X
                y_sample = y
                oob_indices = np.array([])

            # 트리 학습
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=rng.integers(0, 2**31) if self.random_state is not None else None
            )
            tree.fit(X_sample, y_sample)

            # 트리 저장
            self.estimators_.append(tree)
            all_importances.append(tree.feature_importances_)

            # OOB 예측 업데이트
            if self.oob_score and len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                oob_predictions_sum[oob_indices] += oob_pred
                oob_predictions_count[oob_indices] += 1

            # 학습 과정 기록
            train_pred = tree.predict(X_sample)
            train_mse = np.mean((y_sample - train_pred) ** 2)

            self.training_history_.append({
                'tree_idx': m + 1,
                'train_mse': train_mse,
                'tree_depth': tree.get_depth(),
                'tree_n_leaves': tree.get_n_leaves(),
                'n_oob_samples': len(oob_indices) if self.oob_score else 0
            })

            # 진행 상황 출력
            if self.verbose > 0 and (m + 1) % max(1, self.n_estimators // 10) == 0:
                print(f"트리 {m + 1}/{self.n_estimators} 완료")

        # 피처 중요도 계산 (평균)
        self.feature_importances_ = np.mean(all_importances, axis=0)

        # OOB 점수 계산
        if self.oob_score:
            # OOB 예측을 받은 샘플만 사용
            valid_oob = oob_predictions_count > 0

            if np.sum(valid_oob) > 0:
                self.oob_prediction_ = np.zeros(n_samples)
                self.oob_prediction_[valid_oob] = (
                    oob_predictions_sum[valid_oob] / oob_predictions_count[valid_oob]
                )

                # R² 점수 계산
                y_valid = y[valid_oob]
                oob_valid = self.oob_prediction_[valid_oob]

                ss_res = np.sum((y_valid - oob_valid) ** 2)
                ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)

                self.oob_score_ = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                if self.verbose > 0:
                    print(f"OOB R² Score: {self.oob_score_:.4f}")
            else:
                self.oob_score_ = None
                self.oob_prediction_ = None

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        예측 수행 (모든 트리 예측의 평균)

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

        # 모든 트리의 예측 수집
        predictions = np.array([tree.predict(X) for tree in self.estimators_])

        # 평균 반환
        return np.mean(predictions, axis=0)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        예측의 표준편차 반환 (불확실성 추정)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            예측할 데이터

        Returns
        -------
        std : ndarray of shape (n_samples,)
            각 샘플의 예측 표준편차
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return np.std(predictions, axis=0)

    def predict_with_uncertainty(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        예측과 함께 불확실성 구간 반환

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            예측할 데이터

        Returns
        -------
        y_pred : ndarray
            예측값 (평균)
        lower : ndarray
            2.5 백분위수
        upper : ndarray
            97.5 백분위수
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = np.array([tree.predict(X) for tree in self.estimators_])

        y_pred = np.mean(predictions, axis=0)
        lower = np.percentile(predictions, 2.5, axis=0)
        upper = np.percentile(predictions, 97.5, axis=0)

        return y_pred, lower, upper

    def staged_predict(self, X: np.ndarray) -> np.ndarray:
        """
        각 트리 추가 후의 예측 반환 (수렴 분석용)

        Returns
        -------
        predictions : ndarray of shape (n_estimators, n_samples)
            각 단계에서의 누적 평균 예측값
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        n_estimators = len(self.estimators_)

        staged_predictions = np.zeros((n_estimators, n_samples))

        cumulative_sum = np.zeros(n_samples)
        for i, tree in enumerate(self.estimators_):
            cumulative_sum += tree.predict(X)
            staged_predictions[i] = cumulative_sum / (i + 1)

        return staged_predictions

    def get_oob_score(self) -> Optional[float]:
        """OOB R² 점수 반환"""
        return self.oob_score_

    def __repr__(self) -> str:
        if len(self.estimators_) == 0:
            return "RandomForestRegressor(not fitted)"

        oob_str = f", oob_score={self.oob_score_:.4f}" if self.oob_score_ is not None else ""

        return (
            f"RandomForestRegressor("
            f"n_estimators={len(self.estimators_)}, "
            f"max_depth={self.max_depth}"
            f"{oob_str})"
        )
