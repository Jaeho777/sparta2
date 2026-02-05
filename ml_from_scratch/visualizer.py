"""
ML Visualizer - 머신러닝 알고리즘 시각화 도구
=============================================

각 알고리즘의 학습 과정과 내부 동작을 시각화합니다.

주요 기능:
- 결정 트리 구조 시각화
- 부스팅 학습 곡선
- 앙상블 예측 수렴 과정
- 피처 중요도 비교
- 잔차 분석

Author: ML From Scratch Project
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List, Dict, Tuple, Any
import warnings


class MLVisualizer:
    """
    머신러닝 알고리즘 시각화 클래스

    Parameters
    ----------
    figsize : tuple, default=(12, 8)
        기본 Figure 크기

    style : str, default='seaborn-v0_8-whitegrid'
        Matplotlib 스타일

    dpi : int, default=100
        Figure DPI
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        style: Optional[str] = None,
        dpi: int = 100
    ):
        self.figsize = figsize
        self.style = style
        self.dpi = dpi

        # 스타일 설정
        if self.style:
            try:
                plt.style.use(self.style)
            except OSError:
                pass  # 스타일을 찾을 수 없으면 기본값 사용

        # 색상 팔레트
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'neutral': '#3B3B3B',
            'train': '#2E86AB',
            'val': '#F18F01',
            'test': '#C73E1D'
        }

    def plot_decision_tree(
        self,
        tree,
        feature_names: Optional[List[str]] = None,
        max_depth: int = 4,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Decision Tree Structure"
    ) -> plt.Figure:
        """
        결정 트리 구조 시각화

        Parameters
        ----------
        tree : DecisionTreeRegressor
            시각화할 트리
        feature_names : list, optional
            피처 이름 리스트
        max_depth : int
            표시할 최대 깊이
        figsize : tuple, optional
            Figure 크기
        title : str
            그래프 제목

        Returns
        -------
        fig : matplotlib.Figure
        """
        if tree.root_ is None:
            raise ValueError("트리가 학습되지 않았습니다.")

        fig, ax = plt.subplots(figsize=figsize or (14, 10), dpi=self.dpi)

        # 트리 구조 추출
        tree_dict = tree.export_tree_structure()

        # 노드 위치 계산
        positions = self._calculate_tree_positions(tree_dict, max_depth)

        # 노드와 엣지 그리기
        self._draw_tree_nodes(ax, tree_dict, positions, feature_names, max_depth)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig

    def _calculate_tree_positions(
        self,
        node: Dict,
        max_depth: int,
        x: float = 0.5,
        y: float = 0.95,
        x_offset: float = 0.25,
        depth: int = 0,
        positions: Optional[Dict] = None
    ) -> Dict:
        """트리 노드 위치 계산"""
        if positions is None:
            positions = {}

        node_id = id(node)
        positions[node_id] = (x, y, node)

        if depth >= max_depth or node.get('is_leaf', True):
            return positions

        # 자식 노드 위치
        y_child = y - 0.15
        x_offset_child = x_offset / 2

        if 'left' in node:
            self._calculate_tree_positions(
                node['left'], max_depth, x - x_offset, y_child,
                x_offset_child, depth + 1, positions
            )

        if 'right' in node:
            self._calculate_tree_positions(
                node['right'], max_depth, x + x_offset, y_child,
                x_offset_child, depth + 1, positions
            )

        return positions

    def _draw_tree_nodes(
        self,
        ax: plt.Axes,
        node: Dict,
        positions: Dict,
        feature_names: Optional[List[str]],
        max_depth: int,
        depth: int = 0
    ):
        """트리 노드와 엣지 그리기"""
        if depth > max_depth:
            return

        node_id = id(node)
        if node_id not in positions:
            return

        x, y, _ = positions[node_id]

        # 노드 색상 (깊이에 따라)
        cmap = plt.cm.Blues
        color = cmap(0.3 + 0.5 * (1 - depth / max_depth))

        if node.get('is_leaf', True):
            color = plt.cm.Greens(0.6)

        # 노드 박스
        bbox = dict(
            boxstyle='round,pad=0.3',
            facecolor=color,
            edgecolor='gray',
            alpha=0.9
        )

        # 노드 텍스트
        if node.get('is_leaf', True):
            text = f"값: {node['value']:.2f}\n샘플: {node['n_samples']}"
        else:
            feat_idx = node.get('feature_idx', 0)
            feat_name = feature_names[feat_idx] if feature_names else f"X{feat_idx}"
            threshold = node.get('threshold', 0)
            text = f"{feat_name}\n≤ {threshold:.2f}\n샘플: {node['n_samples']}"

        ax.text(x, y, text, ha='center', va='center',
                fontsize=8, bbox=bbox)

        # 자식 노드 연결
        if not node.get('is_leaf', True) and depth < max_depth:
            if 'left' in node:
                left_id = id(node['left'])
                if left_id in positions:
                    x_left, y_left, _ = positions[left_id]
                    ax.plot([x, x_left], [y - 0.03, y_left + 0.03],
                           'k-', linewidth=1, alpha=0.7)
                    ax.text((x + x_left) / 2 - 0.02, (y + y_left) / 2,
                           'T', fontsize=7, color='green')

            if 'right' in node:
                right_id = id(node['right'])
                if right_id in positions:
                    x_right, y_right, _ = positions[right_id]
                    ax.plot([x, x_right], [y - 0.03, y_right + 0.03],
                           'k-', linewidth=1, alpha=0.7)
                    ax.text((x + x_right) / 2 + 0.02, (y + y_right) / 2,
                           'F', fontsize=7, color='red')

        # 재귀적으로 자식 그리기
        if 'left' in node:
            self._draw_tree_nodes(ax, node['left'], positions, feature_names, max_depth, depth + 1)
        if 'right' in node:
            self._draw_tree_nodes(ax, node['right'], positions, feature_names, max_depth, depth + 1)

    def plot_boosting_curve(
        self,
        model,
        title: str = "Boosting Learning Curve",
        figsize: Optional[Tuple[int, int]] = None,
        show_validation: bool = True
    ) -> plt.Figure:
        """
        부스팅 모델의 학습 곡선 시각화

        Parameters
        ----------
        model : GradientBoostingRegressor, AdaBoostRegressor, XGBoostRegressor
            학습된 부스팅 모델
        title : str
            그래프 제목
        figsize : tuple, optional
            Figure 크기
        show_validation : bool
            검증 곡선 표시 여부

        Returns
        -------
        fig : matplotlib.Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize or (14, 5), dpi=self.dpi)

        history = model.training_history_

        if not history:
            raise ValueError("학습 이력이 없습니다.")

        iterations = [h['iteration'] for h in history]

        # 1. MSE/RMSE 곡선
        ax1 = axes[0]

        if 'train_mse' in history[0]:
            train_mse = [h['train_mse'] for h in history]
            ax1.plot(iterations, train_mse, label='Train MSE',
                    color=self.colors['train'], linewidth=2)

            if show_validation and 'val_mse' in history[0]:
                val_mse = [h['val_mse'] for h in history]
                ax1.plot(iterations, val_mse, label='Val MSE',
                        color=self.colors['val'], linewidth=2, linestyle='--')

        elif 'avg_loss' in history[0]:  # AdaBoost
            avg_loss = [h['avg_loss'] for h in history]
            ax1.plot(iterations, avg_loss, label='Avg Loss',
                    color=self.colors['primary'], linewidth=2)

        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('MSE / Loss', fontsize=11)
        ax1.set_title('Learning Curve', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 2. 잔차 통계
        ax2 = axes[1]

        if 'residual_mean' in history[0]:
            residual_mean = [h['residual_mean'] for h in history]
            residual_std = [h['residual_std'] for h in history]

            ax2.fill_between(
                iterations,
                np.array(residual_mean) - np.array(residual_std),
                np.array(residual_mean) + np.array(residual_std),
                alpha=0.3, color=self.colors['primary']
            )
            ax2.plot(iterations, residual_mean, label='Residual Mean',
                    color=self.colors['primary'], linewidth=2)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

            ax2.set_xlabel('Iteration', fontsize=11)
            ax2.set_ylabel('Residual', fontsize=11)
            ax2.set_title('Residual Statistics', fontsize=12, fontweight='bold')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)

        elif 'estimator_weight' in history[0]:  # AdaBoost
            weights = [h['estimator_weight'] for h in history]
            ax2.bar(iterations, weights, color=self.colors['secondary'], alpha=0.7)
            ax2.set_xlabel('Iteration', fontsize=11)
            ax2.set_ylabel('Estimator Weight', fontsize=11)
            ax2.set_title('Estimator Weights (log(1/β))', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_feature_importance(
        self,
        models: Dict[str, Any],
        feature_names: Optional[List[str]] = None,
        top_k: int = 15,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Feature Importance Comparison"
    ) -> plt.Figure:
        """
        여러 모델의 피처 중요도 비교

        Parameters
        ----------
        models : dict
            {모델명: 모델객체} 딕셔너리
        feature_names : list, optional
            피처 이름 리스트
        top_k : int
            표시할 상위 피처 수
        figsize : tuple, optional
            Figure 크기
        title : str
            그래프 제목

        Returns
        -------
        fig : matplotlib.Figure
        """
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=figsize or (5 * n_models, 8), dpi=self.dpi)

        if n_models == 1:
            axes = [axes]

        colors = plt.cm.Set2(np.linspace(0, 1, n_models))

        for idx, (name, model) in enumerate(models.items()):
            ax = axes[idx]

            if model.feature_importances_ is None:
                ax.text(0.5, 0.5, 'Not fitted', ha='center', va='center')
                continue

            importances = model.feature_importances_

            if feature_names is None:
                feature_names_local = [f'Feature {i}' for i in range(len(importances))]
            else:
                feature_names_local = feature_names

            # 상위 k개 선택
            indices = np.argsort(importances)[::-1][:top_k]

            ax.barh(
                range(len(indices)),
                importances[indices],
                color=colors[idx],
                alpha=0.8
            )
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names_local[i] for i in indices])
            ax.invert_yaxis()
            ax.set_xlabel('Importance', fontsize=10)
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_ensemble_convergence(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        sample_indices: Optional[List[int]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Ensemble Prediction Convergence"
    ) -> plt.Figure:
        """
        앙상블 예측의 수렴 과정 시각화

        Parameters
        ----------
        model : RandomForestRegressor or boosting model
            학습된 앙상블 모델
        X : ndarray
            입력 데이터
        y : ndarray
            실제 타겟값
        sample_indices : list, optional
            시각화할 샘플 인덱스
        figsize : tuple, optional
            Figure 크기
        title : str
            그래프 제목

        Returns
        -------
        fig : matplotlib.Figure
        """
        # staged_predict 메서드 확인
        if not hasattr(model, 'staged_predict'):
            raise ValueError("모델에 staged_predict 메서드가 없습니다.")

        staged_preds = model.staged_predict(X)
        n_stages = staged_preds.shape[0]

        if sample_indices is None:
            # 오차가 다양한 5개 샘플 선택
            final_errors = np.abs(staged_preds[-1] - y)
            percentiles = [0, 25, 50, 75, 100]
            sample_indices = [
                np.argmin(np.abs(final_errors - np.percentile(final_errors, p)))
                for p in percentiles
            ]

        fig, axes = plt.subplots(2, 1, figsize=figsize or (12, 8), dpi=self.dpi)

        # 1. 개별 샘플의 예측 수렴
        ax1 = axes[0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(sample_indices)))

        for idx, sample_idx in enumerate(sample_indices):
            ax1.plot(
                range(n_stages),
                staged_preds[:, sample_idx],
                color=colors[idx],
                alpha=0.7,
                label=f'Sample {sample_idx}'
            )
            ax1.axhline(y=y[sample_idx], color=colors[idx],
                       linestyle='--', alpha=0.5)

        ax1.set_xlabel('Number of Estimators', fontsize=11)
        ax1.set_ylabel('Prediction', fontsize=11)
        ax1.set_title('Individual Sample Predictions', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 2. 전체 MSE 수렴
        ax2 = axes[1]
        mse_history = [np.mean((staged_preds[i] - y) ** 2) for i in range(n_stages)]

        ax2.plot(range(n_stages), mse_history,
                color=self.colors['primary'], linewidth=2)
        ax2.fill_between(range(n_stages), mse_history,
                        alpha=0.2, color=self.colors['primary'])

        ax2.set_xlabel('Number of Estimators', fontsize=11)
        ax2.set_ylabel('MSE', fontsize=11)
        ax2.set_title('Overall MSE Convergence', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_residual_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Residual Analysis"
    ) -> plt.Figure:
        """
        잔차 분석 시각화

        Parameters
        ----------
        y_true : ndarray
            실제 값
        y_pred : ndarray
            예측 값
        figsize : tuple, optional
            Figure 크기
        title : str
            그래프 제목

        Returns
        -------
        fig : matplotlib.Figure
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=figsize or (12, 10), dpi=self.dpi)

        # 1. 잔차 vs 예측값
        ax1 = axes[0, 0]
        ax1.scatter(y_pred, residuals, alpha=0.5, color=self.colors['primary'])
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel('Predicted Values', fontsize=10)
        ax1.set_ylabel('Residuals', fontsize=10)
        ax1.set_title('Residuals vs Predicted', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. 잔차 히스토그램
        ax2 = axes[0, 1]
        ax2.hist(residuals, bins=30, color=self.colors['secondary'],
                alpha=0.7, edgecolor='white')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax2.axvline(x=np.mean(residuals), color='blue', linestyle='-',
                   linewidth=1, label=f'Mean: {np.mean(residuals):.2f}')
        ax2.set_xlabel('Residuals', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('Residual Distribution', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 실제 vs 예측
        ax3 = axes[1, 0]
        ax3.scatter(y_true, y_pred, alpha=0.5, color=self.colors['primary'])
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        ax3.set_xlabel('Actual Values', fontsize=10)
        ax3.set_ylabel('Predicted Values', fontsize=10)
        ax3.set_title('Actual vs Predicted', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Q-Q Plot
        ax4 = axes[1, 1]
        from scipy import stats
        try:
            stats.probplot(residuals, dist="norm", plot=ax4)
            ax4.set_title('Q-Q Plot (Normality Check)', fontsize=11, fontweight='bold')
        except ImportError:
            # scipy가 없으면 정규화된 잔차 표시
            sorted_residuals = np.sort(residuals)
            theoretical = np.random.normal(np.mean(residuals), np.std(residuals), len(residuals))
            theoretical = np.sort(theoretical)
            ax4.scatter(theoretical, sorted_residuals, alpha=0.5)
            ax4.plot([theoretical.min(), theoretical.max()],
                    [theoretical.min(), theoretical.max()], 'r--')
            ax4.set_xlabel('Theoretical Quantiles', fontsize=10)
            ax4.set_ylabel('Sample Quantiles', fontsize=10)
            ax4.set_title('Q-Q Plot (Normality Check)', fontsize=11, fontweight='bold')

        ax4.grid(True, alpha=0.3)

        # 통계 정보 추가
        stats_text = (
            f"Mean: {np.mean(residuals):.4f}\n"
            f"Std: {np.std(residuals):.4f}\n"
            f"RMSE: {np.sqrt(np.mean(residuals**2)):.4f}\n"
            f"MAE: {np.mean(np.abs(residuals)):.4f}"
        )
        fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = ['RMSE', 'MAE', 'MAPE'],
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Model Comparison"
    ) -> plt.Figure:
        """
        모델 성능 비교 시각화

        Parameters
        ----------
        results : dict
            {모델명: {메트릭명: 값}} 형태의 딕셔너리
        metrics : list
            비교할 메트릭 리스트
        figsize : tuple, optional
            Figure 크기
        title : str
            그래프 제목

        Returns
        -------
        fig : matplotlib.Figure
        """
        models = list(results.keys())
        n_metrics = len(metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=figsize or (5 * n_metrics, 6), dpi=self.dpi)

        if n_metrics == 1:
            axes = [axes]

        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            values = [results[m].get(metric, 0) for m in models]

            bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='white')

            # 값 표시
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_prediction_timeline(
        self,
        dates: np.ndarray,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        train_end_date: Optional[Any] = None,
        val_end_date: Optional[Any] = None,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Prediction Timeline"
    ) -> plt.Figure:
        """
        시계열 예측 결과 시각화

        Parameters
        ----------
        dates : ndarray
            날짜 인덱스
        y_true : ndarray
            실제 값
        predictions : dict
            {모델명: 예측값} 딕셔너리
        train_end_date : optional
            훈련 종료 날짜
        val_end_date : optional
            검증 종료 날짜
        figsize : tuple, optional
            Figure 크기
        title : str
            그래프 제목

        Returns
        -------
        fig : matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=figsize or (14, 6), dpi=self.dpi)

        # 실제 값
        ax.plot(dates, y_true, 'k-', linewidth=2, label='Actual', alpha=0.8)

        # 예측값들
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
        for (name, pred), color in zip(predictions.items(), colors):
            ax.plot(dates, pred, '--', linewidth=1.5, label=name,
                   color=color, alpha=0.7)

        # 영역 구분
        if train_end_date is not None:
            ax.axvline(x=train_end_date, color='gray', linestyle=':',
                      linewidth=1, alpha=0.7)
            ax.text(train_end_date, ax.get_ylim()[1], ' Train→Val',
                   fontsize=9, color='gray')

        if val_end_date is not None:
            ax.axvline(x=val_end_date, color='gray', linestyle=':',
                      linewidth=1, alpha=0.7)
            ax.text(val_end_date, ax.get_ylim()[1], ' Val→Test',
                   fontsize=9, color='gray')

        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def save_figure(
        self,
        fig: plt.Figure,
        filepath: str,
        dpi: Optional[int] = None
    ):
        """Figure 저장"""
        fig.savefig(filepath, dpi=dpi or self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Figure saved: {filepath}")
