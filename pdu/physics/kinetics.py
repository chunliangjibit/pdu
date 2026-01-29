"""
爆轰动力学模块 V10.2

实现基于专家反馈的高保真物理模型：
1. Miller V2 两阶段铝粉燃烧动力学（诱导期 + 扩散控制）
2. NM 密度依赖活化能（双反应机制）
3. Shaw-Johnson 碳团簇凝结动力学
"""

import jax
import jax.numpy as jnp
from typing import Dict, Optional

# 物理常数
R_GAS = 8.314  # J/(mol·K)
K_BOLTZMANN = 1.380649e-23  # J/K


@jax.jit
def compute_miller_al_reaction_degree_v2(
    P: float, 
    T: float, 
    D0: float = 18e-6,           # 颗粒直径 (m)
    residence_time: float = 1e-6, # 驻留时间 (s)
    T_melt_oxide: float = 2345.0, # Al2O3 熔点 (K)
    E_a_ign: float = 120e3,      # 点火活化能 (J/mol)
    D_eff_base: float = 1e-5,    # 基础扩散系数 (m^2/s)
    k_burn: float = 5.0e4        # 燃烧速率常数
) -> float:
    """
    Miller V2 两阶段精确铝粉燃烧模型
    
    [物理基础]:
    阶段 1 (诱导期): 氧化壳加热至熔化
        - 诱导时间 τ_ign = (D0²/12D_eff) * exp(E_a / RT)
        - 仅当 T > T_melt_oxide 时启动燃烧
    
    阶段 2 (扩散控制燃烧):
        - dα/dt = 3D_eff/r² * (1-α)^(1/3) * [Ox]^n
        - 简化为: α = 1 - exp(-k * P^m * t_burn)
    
    Returns:
        reaction_degree: 反应度 λ ∈ [0, 1]
    """
    # 安全值处理
    T_safe = jnp.maximum(T, 300.0)
    P_safe = jnp.maximum(P, 0.1)  # GPa
    r0 = D0 / 2.0  # 半径
    
    # === 阶段 1: 诱导期计算 ===
    # 有效扩散系数 (压力增强)
    D_eff = D_eff_base * (P_safe / 10.0) ** 0.5
    
    # 诱导时间: τ_ign = (r0²/3D_eff) * exp(E_a/RT)
    tau_ign = (r0**2 / (3.0 * D_eff)) * jnp.exp(E_a_ign / (R_GAS * T_safe))
    tau_ign = jnp.clip(tau_ign, 1e-9, 1e-3)  # 限制在 ns ~ ms 范围
    
    # 点火条件: T 必须超过 Al2O3 熔点
    is_ignited = T_safe > T_melt_oxide
    
    # === 阶段 2: 扩散控制燃烧 ===
    # 有效燃烧时间 (扣除诱导期)
    t_burn = jnp.maximum(residence_time - tau_ign, 0.0)
    
    # 燃烧速率 (压力和粒径依赖)
    # rate = k * P^m / r²
    p_factor = (P_safe / 20.0) ** 1.0  # 20 GPa 归一化
    r_factor = (9e-6 / r0) ** 2        # 9 μm 归一化
    rate = k_burn * p_factor * r_factor
    
    # 反应度演化
    alpha = 1.0 - jnp.exp(-rate * t_burn)
    
    # 未点火则反应度为零
    alpha = jnp.where(is_ignited, alpha, 0.0)
    
    return jnp.clip(alpha, 0.0, 1.0)


# 兼容旧 API
@jax.jit
def compute_miller_al_reaction_degree(
    P: float, 
    T: float, 
    D0: float = 18e-6, 
    residence_time: float = 1e-6,
    k_const: float = 2.0e5,
    m_exp: float = 1.0,
    n_exp: float = 0.5
) -> float:
    """
    [V10.1 兼容接口] - 内部调用 V2 模型
    """
    return compute_miller_al_reaction_degree_v2(P, T, D0, residence_time)


@jax.jit
def compute_nm_decomposition_rate(
    rho: float,
    T: float,
    P: float,
    rho_threshold: float = 1.4  # g/cm³ 机制切换阈值
) -> float:
    """
    NM 密度依赖双机制分解速率
    
    [物理基础]:
    - 低密度 (ρ < 1.4): C-N 键断裂机制
        CH3NO2 → CH3 + NO2
        E_a ≈ 250 kJ/mol
    
    - 高密度 (ρ > 1.4): 双分子/质子转移机制
        CH3NO2 → CH2NO2⁻ + H⁺
        E_a ≈ 150 kJ/mol (压力降低活化能)
    
    Returns:
        rate_factor: 相对反应速率因子 ∈ [0.1, 10.0]
    """
    T_safe = jnp.maximum(T, 300.0)
    
    # 低密度机制参数
    E_a_low = 250e3   # J/mol
    A_low = 1e16      # 指前因子
    
    # 高密度机制参数 (活化能降低)
    E_a_high = 150e3  # J/mol
    A_high = 1e14
    
    # 密度平滑切换 (sigmoid)
    sigma = 0.1
    w_high = 1.0 / (1.0 + jnp.exp(-(rho - rho_threshold) / sigma))
    w_low = 1.0 - w_high
    
    # 各机制速率
    rate_low = A_low * jnp.exp(-E_a_low / (R_GAS * T_safe))
    rate_high = A_high * jnp.exp(-E_a_high / (R_GAS * T_safe))
    
    # 混合速率
    rate_mixed = w_low * rate_low + w_high * rate_high
    
    # 归一化为相对因子
    rate_ref = 1e10  # 参考速率
    factor = rate_mixed / rate_ref
    
    return jnp.clip(factor, 0.1, 10.0)


@jax.jit
def compute_sj_carbon_energy_delay(
    T: float,
    V: float,
    carbon_fraction: float = 0.1,
    duration: float = 1e-6
) -> float:
    """
    Shaw-Johnson V2 碳团簇凝结动力学
    
    [物理基础]:
    碳团簇成核与生长遵循经典成核理论:
    - τ_nucleation ~ exp(ΔG*/kT) 
    - 高温加速成核，低密度延迟凝结
    
    Returns:
        energy_factor: 能量释放因子 ∈ [0.1, 1.0]
    """
    T_safe = jnp.maximum(T, 1000.0)
    
    # 碳凝结激活能 (来自 DFT 计算)
    E_a_carbon = 80e3  # J/mol
    
    # 成核时间常数
    tau_base = 1e-6  # 基础时间 1 μs
    tau = tau_base * jnp.exp(E_a_carbon / (R_GAS * T_safe))
    tau = jnp.clip(tau, 1e-9, 1e-3)
    
    # 指数逼近能量释放
    factor = 1.0 - jnp.exp(-duration / tau)
    
    # 碳含量加权
    factor = factor * (1.0 - 0.5 * carbon_fraction)
    
    return jnp.clip(factor, 0.1, 1.0)


@jax.jit
def compute_effective_detonation_energy(
    Q_ideal: float,
    al_reaction_degree: float,
    carbon_delay_factor: float,
    combustion_efficiency: float = 1.0
) -> float:
    """
    计算有效爆轰能量
    
    Q_eff = Q_base + λ_Al * Q_Al * η + f_carbon * Q_carbon
    
    Returns:
        Q_effective: 有效爆热 (MJ/kg)
    """
    # 铝燃烧贡献 (Al + 0.75 O2 → 0.5 Al2O3, ΔH = -31 MJ/kg Al)
    Q_al_combustion = 31.0  # MJ/kg Al
    
    # 碳凝结延迟校正
    Q_carbon_correction = 0.5  # MJ/kg 估计
    
    # 有效能量
    Q_eff = Q_ideal * combustion_efficiency
    Q_eff = Q_eff + al_reaction_degree * Q_al_combustion * 0.2  # 假设 20% Al 含量
    Q_eff = Q_eff * carbon_delay_factor
    
    return Q_eff

