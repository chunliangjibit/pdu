"""
逆向设计模块

提供配方优化功能：
- optimizer: 逆向优化器
- loss: 损失函数
- constraints: 约束定义
"""

from pdu.inverse.optimizer import (
    optimize_recipe,
    OptimizationResult,
    RecipeConstraints,
    multi_objective_loss
)
from pdu.inverse.constraints import check_constraints

__all__ = [
    "optimize_recipe",
    "OptimizationResult",
    "RecipeConstraints",
    "multi_objective_loss",
    "check_constraints",
]
