"""
AI 模块

提供 AI 加速功能：
- warmstart: Tiny-MLP 热启动网络
- projection: 原子守恒投影层
"""

from pdu.ai.warmstart import WarmStartMLP, MLPParams, predict_initial_state
from pdu.ai.projection import (
    ConservationProjector,
    project_to_conservation,
    project_log_space,
    project_with_gradient
)

__all__ = [
    "WarmStartMLP",
    "MLPParams",
    "predict_initial_state",
    "ConservationProjector",
    "project_to_conservation",
    "project_log_space",
    "project_with_gradient",
]
