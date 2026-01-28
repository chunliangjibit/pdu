"""
损失函数模块

定义多目标优化的损失函数。
"""

import jax.numpy as jnp
from typing import Dict


def multi_objective_loss(
    performance: Dict[str, float],
    targets: Dict[str, float],
    weights: Dict[str, float] = None
) -> float:
    """多目标加权损失函数
    
    L = Σ w_i * ((y_i - target_i) / scale_i)²
    
    Args:
        performance: 预测性能字典
        targets: 目标性能字典
        weights: 权重字典
        
    Returns:
        总损失值
    """
    from pdu.inverse.optimizer import multi_objective_loss as _loss
    return _loss(performance, targets, weights)


def detonation_performance_loss(
    D_pred: float,
    P_pred: float,
    D_target: float = None,
    P_target: float = None,
    w_D: float = 1.0,
    w_P: float = 1.0
) -> float:
    """爆轰性能损失
    
    Args:
        D_pred: 预测爆速 (m/s)
        P_pred: 预测爆压 (GPa)
        D_target: 目标爆速
        P_target: 目标爆压
        w_D, w_P: 权重
        
    Returns:
        损失值
    """
    loss = 0.0
    
    if D_target is not None:
        loss += w_D * ((D_pred - D_target) / 1000.0) ** 2
    
    if P_target is not None:
        loss += w_P * ((P_pred - P_target) / 10.0) ** 2
    
    return loss


def sensitivity_penalty(h50: float, h50_min: float = 20.0) -> float:
    """感度惩罚项
    
    当感度过高时 (h50 过低) 添加惩罚。
    
    Args:
        h50: 撞击感度 (cm)
        h50_min: 最小可接受感度
        
    Returns:
        惩罚值
    """
    violation = jnp.maximum(h50_min - h50, 0.0)
    return 10.0 * violation ** 2


def oxygen_balance_penalty(
    OB: float,
    OB_min: float = -30.0,
    OB_max: float = 10.0
) -> float:
    """氧平衡惩罚项
    
    Args:
        OB: 氧平衡 (%)
        OB_min, OB_max: 可接受范围
        
    Returns:
        惩罚值
    """
    penalty = 0.0
    
    if OB < OB_min:
        penalty += ((OB_min - OB) / 10.0) ** 2
    
    if OB > OB_max:
        penalty += ((OB - OB_max) / 10.0) ** 2
    
    return penalty
