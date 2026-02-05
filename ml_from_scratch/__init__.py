"""
ML From Scratch - 머신러닝 알고리즘 직접 구현
=============================================

이 모듈은 주요 머신러닝 알고리즘을 라이브러리 없이 직접 구현합니다.
NumPy만 사용하여 알고리즘의 수학적 원리를 명확히 보여줍니다.

구현된 알고리즘:
- DecisionTreeRegressor: CART 기반 결정 트리
- GradientBoostingRegressor: 잔차 학습 기반 부스팅
- AdaBoostRegressor: 가중치 기반 부스팅
- XGBoostRegressor: 2차 근사 + 정규화 기반 부스팅
- RandomForestRegressor: 배깅 기반 앙상블

Author: ML From Scratch Project
"""

from .decision_tree import DecisionTreeRegressor
from .gradient_boosting import GradientBoostingRegressor
from .adaboost import AdaBoostRegressor
from .xgboost import XGBoostRegressor
from .random_forest import RandomForestRegressor
from .visualizer import MLVisualizer

__all__ = [
    'DecisionTreeRegressor',
    'GradientBoostingRegressor',
    'AdaBoostRegressor',
    'XGBoostRegressor',
    'RandomForestRegressor',
    'MLVisualizer'
]

__version__ = '1.0.0'
