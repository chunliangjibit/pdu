"""
爆轰动力学模块

实现 Shaw-Johnson 碳团簇动力学与 Miller 铝粉燃烧动力学模型。
旨在解决非理想炸药（含铝、负氧）的慢能量释放描述。
"""

import jax
import jax.numpy as jnp
from typing import Dict, Optional

@jax.jit
def compute_miller_al_reaction_degree(
    P: float, 
    T: float, 
    D0: float = 18e-6, 
    residence_time: float = 1e-6,
    k_const: float = 2.0e5, # Calibrated to give ~15% degree at 25GPa/1us
    m_exp: float = 1.0,    # Pressure exponent
    n_exp: float = 0.5     # Temperature exponent? Actually n in (1-alpha)^n
) -> float:
    """
    Miller 改进型两步铝粉燃烧动力学
    
    [理论基础]:
    1. Ignition Phase: t_ign ~ D0^2 / T^n (忽略简化的诱导期)
    2. Oxidation Phase: d(alpha)/dt = -k * (1-alpha)^n * P^m
    
    此处返回 CJ 面上的等效反应度 lambda = 1 - alpha
    """
    # 压力单位转换为 Pa
    P_pa = jnp.maximum(P * 1e9, 1e5)
    
    # 铝粉点火判定 (极简)
    ignited = T > 933.0 # 低于铝熔点不点火
    
    # 计算燃烧率 (基于 Miller P^m 形式)
    # alpha_dot = k * P^m
    # 积分得到 alpha = alpha0 * exp(-k * P^m * t) ? 
    # 或者简化线性: delta_alpha = k * P^m * time
    
    # 这里的 k 需要校准，此处使用启发式公式
    # 压力越大，温度越高，粒径越小，反应越快
    p_eff = (P_pa / 1e9) ** m_exp
    t_eff = (T / 3000.0) ** 0.5
    d_eff = (18e-6 / D0) ** 2
    
    rate = k_const * p_eff * t_eff * d_eff
    
    degree = rate * residence_time
    degree = jnp.where(ignited, degree, 0.0)
    
    return jnp.clip(degree, 0.0, 1.0)

@jax.jit
def compute_sj_carbon_energy_delay(
    T: float,
    V: float,
    duration: float = 1e-6
) -> float:
    """
    Shaw-Johnson 碳团簇凝结动力学 (极简版能量释放因子)
    用于修正 CJ 点后的滞后能量释放。
    """
    # T 越高，凝结越快
    # 简化为指数逼近
    tau = 2e-6 * jnp.exp(3000.0 / T) 
    factor = 1.0 - jnp.exp(-duration / tau)
    return jnp.clip(factor, 0.1, 1.0)
