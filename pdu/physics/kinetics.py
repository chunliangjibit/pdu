"""
爆轰动力学模块 V10.4
基于专家反馈技术白皮书的高保真物理模型：
1. Miller-PDU V4 铝粉燃烧动力学
   - 剪切剥离效应 (Shear Stripping) - V10.4 P0级改进
   - 氧化帽效应 (Oxide Cap Effect)
   - Damköhler 自适应混合燃烧律
2. 有效爆热物理化截止 (Freeze-Out Temperature)
3. NM 密度依赖活化能（双反应机制）
4. Shaw-Johnson 碳团簇凝结动力学
"""

import jax
import jax.numpy as jnp
from typing import Dict, Optional

# 物理常数
R_GAS = 8.314  # J/(mol·K)
K_BOLTZMANN = 1.380649e-23  # J/K


# ==============================================================================
# V10.3 新增: 氧化帽效应 (Oxide Cap Effect)
# ==============================================================================

@jax.jit
def compute_oxide_cap_coverage(
    reaction_degree: float,
    t_burn: float,
    D0: float = 18e-6,
    cap_growth_rate: float = 0.12,
    max_coverage: float = 0.70
) -> float:
    """
    氧化帽覆盖率演化 (V10.3 Miller-PDU)
    
    [物理基础]:
    燃烧产物 Al₂O₃ 部分凝结回落到颗粒表面，形成"氧化帽"。
    覆盖率随反应进程和时间演化：
    
    θ_cap = min(k_cap * α * sqrt(t_burn / τ_ref), θ_max)
    
    Args:
        reaction_degree: 当前反应度 α ∈ [0, 1]
        t_burn: 有效燃烧时间 (s)
        D0: 颗粒直径 (m)
        cap_growth_rate: 覆盖增长速率常数
        max_coverage: 最大覆盖率 (通常 0.6-0.7)
    
    Returns:
        theta_cap: 氧化帽覆盖率 ∈ [0, max_coverage]
    """
    # 参考时间尺度 (与颗粒尺寸相关)
    tau_ref = (D0 / 18e-6) ** 2 * 1e-6  # 18μm 颗粒约 1μs
    
    # 时间因子 (平方根关系，早期快速增长后趋于饱和)
    time_factor = jnp.sqrt(jnp.maximum(t_burn / tau_ref, 0.0))
    
    # 覆盖率 = 增长率 × 反应度 × 时间因子
    theta_cap = cap_growth_rate * reaction_degree * time_factor
    
    return jnp.clip(theta_cap, 0.0, max_coverage)


@jax.jit
def compute_effective_surface_area_factor(
    theta_cap: float
) -> float:
    """
    有效活性表面积因子
    
    [物理基础]:
    氧化帽覆盖减少了铝蒸气扩散的有效表面积。
    
    S_eff / S_total = 1 - θ_cap
    
    Returns:
        surface_factor: 有效表面积比例 ∈ [0.3, 1.0]
    """
    return jnp.maximum(1.0 - theta_cap, 0.3)


# ==============================================================================
# V10.3 新增: Damköhler 自适应燃烧律
# ==============================================================================

@jax.jit
def compute_damkohler_number(
    D0: float,
    P: float,
    T: float,
    D_eff_base: float = 1e-5,
    E_a_chem: float = 80e3
) -> float:
    """
    Damköhler 数计算 (V10.3)
    
    [物理基础]:
    Da = τ_diff / τ_chem
    
    τ_diff = r₀² / D_eff (扩散时间尺度)
    τ_chem = 1 / k_chem (化学时间尺度)
    k_chem = A * exp(-E_a / RT)
    
    Da >> 1: 扩散控制 (大颗粒/低温)
    Da << 1: 动力学控制 (小颗粒/高温)
    
    Args:
        D0: 颗粒直径 (m)
        P: 压力 (GPa)
        T: 温度 (K)
    
    Returns:
        Da: Damköhler 数 (对数尺度更有意义)
    """
    T_safe = jnp.maximum(T, 300.0)
    r0 = D0 / 2.0
    
    # 有效扩散系数 (压力增强)
    D_eff = D_eff_base * (P / 10.0) ** 0.3
    
    # 扩散时间尺度
    tau_diff = r0 ** 2 / D_eff
    
    # 化学速率常数 (Arrhenius)
    A_chem = 1e8  # 指前因子
    k_chem = A_chem * jnp.exp(-E_a_chem / (R_GAS * T_safe))
    
    # 化学时间尺度
    tau_chem = 1.0 / jnp.maximum(k_chem, 1e-20)
    
    # Damköhler 数
    Da = tau_diff / tau_chem
    
    return jnp.clip(Da, 1e-3, 1e6)


@jax.jit
def compute_adaptive_burn_exponent(
    Da: float,
    n_kinetic: float = 1.5,
    n_diffusion: float = 2.0
) -> float:
    """
    自适应燃烧指数 (V10.3)
    
    [物理基础]:
    - 扩散控制: 燃烧遵循 D² 定律 (n = 2)
    - 动力学控制: 燃烧遵循 D^n 定律 (n ≈ 1.5)
    
    使用 sigmoid 平滑过渡:
    n_eff = n_kin + (n_diff - n_kin) * σ(log₁₀(Da))
    
    Returns:
        n_eff: 有效燃烧指数 ∈ [n_kinetic, n_diffusion]
    """
    # 在 Da=1 处过渡
    log_Da = jnp.log10(jnp.maximum(Da, 1e-10))
    
    # Sigmoid 过渡 (log_Da = 0 时为 0.5)
    transition_width = 1.5  # 过渡宽度 (对数单位)
    weight_diffusion = 1.0 / (1.0 + jnp.exp(-log_Da / transition_width))
    
    n_eff = n_kinetic + (n_diffusion - n_kinetic) * weight_diffusion
    
    return n_eff


# ==============================================================================
# V10.3 新增: 有效爆热物理化截止
# ==============================================================================

@jax.jit
def compute_freeze_out_temperature(
    P: float,
    expansion_rate: float = 1e6,
    E_a_al: float = 120e3,
    A_al: float = 1e12
) -> float:
    """
    冻结温度计算 (V10.3)
    
    [物理基础]:
    当反应速率等于膨胀速率时，化学反应实际上"冻结"。
    
    k(T_freeze) = expansion_rate
    A * exp(-E_a / RT_freeze) = expansion_rate
    T_freeze = E_a / (R * ln(A / expansion_rate))
    
    Args:
        P: 压力 (GPa), 用于调整活化能
        expansion_rate: 体积膨胀速率 (1/s), 典型值 1e6
    
    Returns:
        T_freeze: 冻结温度 (K), 典型值 1800-2200 K
    """
    # 压力相关的有效活化能 (高压降低势垒)
    P_safe = jnp.maximum(P, 0.1)
    E_a_eff = E_a_al * (1.0 - 0.1 * jnp.log10(P_safe / 1.0))
    E_a_eff = jnp.maximum(E_a_eff, 60e3)  # 下限
    
    # 计算冻结温度
    ratio = A_al / jnp.maximum(expansion_rate, 1.0)
    ln_ratio = jnp.log(jnp.maximum(ratio, 1.0))
    
    T_freeze = E_a_eff / (R_GAS * jnp.maximum(ln_ratio, 0.1))
    
    return jnp.clip(T_freeze, 1500.0, 2500.0)


@jax.jit
def compute_effective_heat_release_factor(
    T_local: float,
    T_freeze: float,
    transition_width: float = 200.0
) -> float:
    """
    有效能量释放因子 (V10.3)
    
    [物理基础]:
    温度低于冻结温度时，反应能量无法有效释放到爆轰波。
    
    η = sigmoid((T_local - T_freeze) / ΔT)
    
    T >> T_freeze: η → 1 (完全释放)
    T << T_freeze: η → 0 (冻结)
    
    Returns:
        eta: 有效释放因子 ∈ [0, 1]
    """
    T_safe = jnp.maximum(T_local, 300.0)
    
    # Sigmoid 过渡
    x = (T_safe - T_freeze) / transition_width
    eta = 1.0 / (1.0 + jnp.exp(-x))
    
    return jnp.clip(eta, 0.05, 1.0)


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


# ==============================================================================
# V10.3 核心: Miller-PDU V3 完整模型
# ==============================================================================

@jax.jit
def compute_miller_al_reaction_degree_v3(
    P: float, 
    T: float, 
    D0: float = 18e-6,           # 颗粒直径 (m)
    residence_time: float = 1e-6, # 驻留时间 (s)
    T_melt_oxide: float = 2345.0, # Al2O3 熔点 (K)
    E_a_ign: float = 120e3,      # 点火活化能 (J/mol)
    D_eff_base: float = 1e-5     # 基础扩散系数 (m^2/s)
) -> float:
    """
    Miller-PDU V3 完整铝粉燃烧模型 (V10.3)
    
    [核心改进]:
    1. 氧化帽效应: 燃烧产物回落导致活性表面积衰减
    2. Damköhler 自适应燃烧律: D² ↔ D^n 平滑过渡
    3. 物理诱导期: 考虑氧化层热穿透
    
    [物理基础]:
    阶段 1 (诱导期): 氧化壳加热至熔化
        - τ_ign = (r0²/3D_eff) * exp(E_a / RT)
        - 仅当 T > T_melt_oxide 时启动燃烧
    
    阶段 2 (混合控制燃烧):
        - Damköhler 数决定控制机制
        - 氧化帽覆盖率限制有效表面积
        - α = 1 - exp(-rate * S_eff * t_burn)
    
    Returns:
        reaction_degree: 反应度 λ ∈ [0, 1]
    """
    # 安全值处理
    T_safe = jnp.maximum(T, 300.0)
    P_safe = jnp.maximum(P, 0.1)  # GPa
    r0 = D0 / 2.0  # 半径
    
    # === 阶段 1: 诱导期计算 ===
    # V10.3: 引入压力相关的有效活化能
    # 高压下活化能显著降低 (机械应力 + 化学侵蚀)
    # 参考: 专家反馈中的"氧化层水合作用"
    E_a_effective = E_a_ign * (1.0 - 0.3 * jnp.log10(P_safe / 1.0))  # 20 GPa 时约 80 kJ/mol
    E_a_effective = jnp.clip(E_a_effective, 50e3, E_a_ign)
    
    # 有效扩散系数 (压力增强更显著)
    # 高压高密度气体扩散更快 (D_eff ~ P)
    D_eff = D_eff_base * (P_safe / 1.0) ** 0.8  # 更强的压力依赖
    
    # 诱导时间: τ_ign = (r0²/3D_eff) * exp(E_a/RT)
    tau_ign_raw = (r0**2 / (3.0 * D_eff)) * jnp.exp(E_a_effective / (R_GAS * T_safe))
    
    # V10.3: 水合效应加速因子 (H2O 与 Al2O3 反应削弱氧化层)
    # 在爆轰产物中 H2O 丰富，加速氧化层破裂
    hydration_factor = 0.1  # 加速 10 倍
    tau_ign = tau_ign_raw * hydration_factor
    tau_ign = jnp.clip(tau_ign, 1e-9, 1e-4)  # 典型值 ns ~ 100 μs
    
    # 点火条件: T 必须超过 Al2O3 熔点
    is_ignited = T_safe > T_melt_oxide
    
    # V10.3: 扩展驻留时间估计
    # 爆轰反应区宽度约 1-10 mm，波速约 7000 m/s，驻留时间 0.1-1.4 μs
    # 但铝后燃可延伸到膨胀区，有效驻留时间更长
    effective_residence = jnp.maximum(residence_time * 10.0, 5e-6)  # 至少 5 μs
    
    # === 阶段 2: 混合控制燃烧 ===
    # 有效燃烧时间 (扣除诱导期)
    t_burn = jnp.maximum(effective_residence - tau_ign, 0.0)
    
    # V10.3: 计算 Damköhler 数和自适应燃烧指数
    Da = compute_damkohler_number(D0, P_safe, T_safe, D_eff_base)
    n_burn = compute_adaptive_burn_exponent(Da)
    
    # 基础燃烧速率 (使用自适应指数)
    # rate ~ P^m / r^n
    p_factor = (P_safe / 20.0) ** 1.0  # 20 GPa 归一化
    r_factor = (9e-6 / r0) ** n_burn   # 自适应指数
    k_burn = 5.0e4  # 基础速率常数
    rate = k_burn * p_factor * r_factor
    
    # 初步反应度 (无氧化帽修正)
    alpha_raw = 1.0 - jnp.exp(-rate * t_burn)
    
    # V10.3: 氧化帽效应修正
    theta_cap = compute_oxide_cap_coverage(alpha_raw, t_burn, D0)
    S_eff_factor = compute_effective_surface_area_factor(theta_cap)
    
    # 修正后的反应度 (活性表面积衰减)
    rate_corrected = rate * S_eff_factor
    alpha = 1.0 - jnp.exp(-rate_corrected * t_burn)
    
    # 未点火则反应度为零
    alpha = jnp.where(is_ignited, alpha, 0.0)
    
    return jnp.clip(alpha, 0.0, 1.0)


@jax.jit
def compute_effective_detonation_energy_v2(
    Q_ideal: float,
    al_reaction_degree: float,
    al_mass_fraction: float,
    T_cj: float,
    P_cj: float,
    carbon_delay_factor: float = 1.0
) -> float:
    """
    V10.3 物理化有效爆轰能量
    
    [核心改进]:
    1. 冻结温度截止: 低温时铝反应能量无法释放
    2. 物理计算替代手动 combustion_efficiency
    
    Args:
        Q_ideal: 理想爆热 (MJ/kg)
        al_reaction_degree: 铝反应度
        al_mass_fraction: 铝质量分数
        T_cj: CJ 温度 (K)
        P_cj: CJ 压力 (GPa)
        carbon_delay_factor: 碳凝结因子
    
    Returns:
        Q_effective: 有效爆热 (MJ/kg)
    """
    # 计算冻结温度
    T_freeze = compute_freeze_out_temperature(P_cj)
    
    # 有效能量释放因子
    eta_release = compute_effective_heat_release_factor(T_cj, T_freeze)
    
    # 铝燃烧贡献 (ΔH = -31 MJ/kg Al)
    Q_al_combustion = 31.0  # MJ/kg Al
    
    # 铝在 CJ 面的有效能量贡献 (受冻结温度限制)
    Q_al_effective = al_reaction_degree * al_mass_fraction * Q_al_combustion * eta_release
    
    # 基础爆热 (不含铝贡献)
    Q_base = Q_ideal * (1.0 - al_mass_fraction)
    
    # 总有效爆热
    Q_eff = Q_base + Q_al_effective
    Q_eff = Q_eff * carbon_delay_factor
    
    return Q_eff

# ==============================================================================
# V10.4 新增: 剪切剥离模型 (Shear Stripping) - P0级改进
# ==============================================================================

@jax.jit
def compute_velocity_slip(
    P: float,
    rho_gas: float
) -> float:
    """
    估算粒子与气体的滑移速度 (V10.4)
    
    [物理基础]:
    爆轰波后气体速度极高 (u_p ~ 1000-2000 m/s)，重粒子(Al)此时速度较低。
    u_slip = u_gas - u_particle
    简化的压力相关经验公式。
    
    Args:
        P: 压力 (GPa)
        rho_gas: 气体密度 (g/cm³)
        
    Returns:
        v_slip: 滑移速度 (m/s)
    """
    # 经验相关式：压力越高，波速越快，滑移越大
    # 参考: P ≈ 20 GPa -> v_slip ≈ 400 m/s
    v_slip = 100.0 * (P / 1.0) ** 0.6 
    return jnp.clip(v_slip, 0.0, 1500.0)


@jax.jit
def compute_weber_number(
    rho_gas: float,
    v_slip: float,
    D0: float,
    sigma_al: float = 0.9
) -> float:
    """
    计算韦伯数 (Weber Number)
    
    We = (ρ_gas * v_slip² * D) / σ
    
    Args:
        rho_gas: 气体密度 (g/cm³) -> 需要转为 kg/m³
        v_slip: 滑移速度 (m/s)
        D0: 颗粒直径 (m)
        sigma_al: 熔融铝/氧化铝 表面张力 (N/m), 典型值 0.8-1.0
        
    Returns:
        We: 无量纲韦伯数
    """
    rho_kg_m3 = rho_gas * 1000.0
    We = (rho_kg_m3 * v_slip**2 * D0) / sigma_al
    return We


@jax.jit
def compute_shear_stripping_factor(
    We: float,
    We_crit: float = 60.0  # 临界韦伯数, 专家建议 50-100
) -> float:
    """
    计算剪切剥离造成的表面积增益因子 (V10.4)
    
    [物理基础]:
    当 We > We_crit 时，Kelvin-Helmholtz 不稳定性导致表面液层剥离/雾化，
    显著增加反应表面积，并移除氧化层阻碍。
    
    Factor = 1 + k * (We - We_crit) / We_crit
    
    Returns:
        stripping_factor: >= 1.0
    """
    # 剥离强度
    excess_We = jnp.maximum(We - We_crit, 0.0)
    
    # 增益系数 k: 剥离效率
    # 专家指出 Tritonal 中剥离显著，需较大增益
    k_strip = 2.0 
    
    stripping_factor = 1.0 + k_strip * (excess_We / We_crit)
    
    # 物理限制: 表面积增加即便在强剪切下也是有限的（颗粒破碎极限）
    return jnp.clip(stripping_factor, 1.0, 10.0)


@jax.jit
def compute_miller_al_reaction_degree_v4(
    P: float, 
    T: float, 
    D0: float = 18e-6,
    residence_time: float = 1e-6,
    T_melt_oxide: float = 2345.0,
    E_a_ign: float = 120e3,
    D_eff_base: float = 1e-5
) -> float:
    """
    Miller-PDU V4 完整模型 (V10.4)
    
    集成:
    1. V10.3 的氧化帽、Da 自适应、压力降活化能、水合效应
    2. V10.4 P0级改进: 剪切剥离 (Shear Stripping)
       - 高压高剪切下，氧化帽被剥离
       - 反应表面积增加
       - 诱导期被旁路
    
    Returns:
        alpha: 反应度 ∈ [0, 1]
    """
    # 1. 基础物理量估算
    # 估算气体密度 (用于 We 计算)
    # PM = rho RT => rho = P M / R T. M_avg ~ 30 g/mol
    rho_gas_est = (P * 1e9 * 0.030) / (R_GAS * T + 1e-6) / 1000.0 # g/cm3
    rho_gas_est = jnp.clip(rho_gas_est, 0.1, 3.0)
    
    # 2. 计算剪切剥离效应
    v_slip = compute_velocity_slip(P, rho_gas_est)
    We = compute_weber_number(rho_gas_est, v_slip, D0)
    stripping_factor = compute_shear_stripping_factor(We)
    
    # === 阶段 1: 诱导期计算 (受剥离影响) ===
    r0 = D0 / 2.0
    T_safe = jnp.maximum(T, 300.0)
    P_safe = jnp.maximum(P, 0.1)
    
    # 压力修正活化能 & 扩散系数 (保留 V10.3 逻辑)
    E_a_effective = E_a_ign * (1.0 - 0.3 * jnp.log10(P_safe / 1.0))
    E_a_effective = jnp.clip(E_a_effective, 50e3, E_a_ign)
    D_eff = D_eff_base * (P_safe / 1.0) ** 0.8
    
    tau_ign_raw = (r0**2 / (3.0 * D_eff)) * jnp.exp(E_a_effective / (R_GAS * T_safe))
    
    # 水合效应
    hydration_factor = 0.1
    tau_ign = tau_ign_raw * hydration_factor
    
    # [V10.4 关键修正]: 强剪切剥离会移除氧化层，急剧缩短甚至消除诱导期
    # 如果 stripping_factor >> 1，说明氧化层不复存在
    stripping_bypass = 1.0 / stripping_factor
    tau_ign_effective = tau_ign * stripping_bypass
    
    # 扩展驻留时间 (保留 V10.3)
    effective_residence = jnp.maximum(residence_time * 10.0, 5e-6)
    
    # === 阶段 2: 燃烧计算 ===
    t_burn = jnp.maximum(effective_residence - tau_ign_effective, 0.0)
    
    # Da 自适应 (保留 V10.3)
    Da = compute_damkohler_number(D0, P_safe, T_safe, D_eff_base)
    n_burn = compute_adaptive_burn_exponent(Da)
    
    # 基础速率
    p_factor = (P_safe / 20.0) ** 1.0
    r_factor = (9e-6 / r0) ** n_burn
    k_burn = 5.0e4
    rate_base = k_burn * p_factor * r_factor
    
    # [V10.4 关键修正]: 反应速率乘以剥离因子 (表面积增大)
    rate_enhanced = rate_base * stripping_factor
    
    # 初始反应度
    alpha_raw = 1.0 - jnp.exp(-rate_enhanced * t_burn)
    
    # 氧化帽效应 (保留 V10.3，但在强剥离下被抑制)
    # 如果 stripping_factor 大，说明没有帽
    theta_cap_raw = compute_oxide_cap_coverage(alpha_raw, t_burn, D0)
    theta_cap_effective = theta_cap_raw / stripping_factor # 剥离减少覆盖
    
    S_eff_factor = compute_effective_surface_area_factor(theta_cap_effective)
    
    # 重新计算最终 alpha
    rate_final = rate_enhanced * S_eff_factor
    
    alpha = 1.0 - jnp.exp(-rate_final * t_burn)
    
    return jnp.clip(alpha, 0.0, 1.0)


# ==============================================================================
# V10.5 新增: 动力学冻结模型 (Thermal Lag) - P0-4 改进
# ==============================================================================

@jax.jit
def compute_oxide_thermal_lag(
    D0: float,
    T_gas: float,
    P_gas: float,
    T_melt_oxide: float = 2327.0,
    n_diameter: float = 1.6  # V10.5 Adaptive Exponent (Expert Feedback)
) -> float:
    """
    计算氧化层受热熔化/破裂的诱导时间 (Induction Time)
    
    [物理基础]:
    微米级铝粉表面覆盖 Al2O3 壳层 (Tm ~ 2300 K)。
    即使气体温度高达 3000-4000 K，热量传导至壳层并使其熔化需要时间。
    t_ind ~ (D^2 / alpha) * f(Bi)
    
    对于 10-50 um 颗粒，tau 通常在 0.1 - 5.0 us 范围。
    而 CJ 反应区时间尺度仅 ~10 ns (0.01 us)。
    
    Args:
        D0: 颗粒直径 (m)
        T_gas: 气体温度 (K)
        n_diameter: 粒径指数 (Default 1.6 for mAl, 1.0 for nAl)
        
    Returns:
        tau_ind: 热滞后时间 (s)
    """
    # 简化物理估算: 
    # tau = k * D0^2
    # 标定: D0=18um -> tau=2.0us (根据圆筒实验经验)
    
    d_ref = 18e-6
    tau_ref = 2.0e-6 
    
    # 温度越高，加热越快 (Arrhenius-like or Linear delta-T)
    # Factor = (Tm - T0) / (T_gas - T0) ? 
    # 简化为幂律修正
    temp_factor = (3500.0 / jnp.maximum(T_gas, 2000.0)) ** 1.0
    
    # 压力越高，传热越快 (Nu ~ Re^0.5 ~ rho^0.5 ~ P^0.5)
    press_factor = (20.0 / jnp.maximum(P_gas, 1.0)) ** 0.3
    
    tau_ind = tau_ref * (D0 / d_ref)**n_diameter * temp_factor * press_factor
    
    return jnp.clip(tau_ind, 1e-9, 1e-4)


@jax.jit
def compute_miller_al_reaction_degree_v5(
    P: float, 
    T: float, 
    D0: float = 18e-6,
    residence_time: float = 1e-6, # 默认驻留时间，爆轰波阵面处极短
    time_scale_cj: float = 1e-8, # CJ 面特征时间 (10 ns)
    E_a_ign: float = 120e3
) -> float:
    """
    Miller-PDU V5 完整模型 (V10.5)
    
    核心改进:
    1. 引入热滞后 (Thermal Lag): 强制 CJ 处反应度归零
    2. 串联剪切剥离 (Shear Stripping): 仅在 t > tau_ind 后激活
    
    流程:
    If t < tau_ind: 
       Reaction = 0 (Frozen / Inert)
    Else: 
       Use V4 Shear Stripping Kinetics
       
    注意: 此函数计算的是 "CJ面时刻" 的反应度。
    由于 CJ 面对应的时间极短 (time_scale_cj ~ 10-50ns)，而 tau_ind ~ us，
    因此预期返回值为 0 或极小。
    
    Returns:
        alpha: 反应度
    """
    # 1. 计算热滞后时间
    tau_ind = compute_oxide_thermal_lag(D0, T, P)
    
    # 2. 判断当前时间点 (CJ 面) 是否超过滞后时间
    # 在 detonation_forward 中，我们评估的是 CJ 状态瞬间
    current_time = time_scale_cj 
    
    # Gate Function: Smooth transition usually better for gradients, but hard gate implies physics
    # 使用 sigmoid 平滑过渡以保持可微性
    # gate = 0 if t < tau, 1 if t > tau
    # steepness k
    gate = 0.5 * (jnp.tanh((current_time - tau_ind) / (tau_ind * 0.1)) + 1.0)
    
    # 3. 计算 V4 动力学 (假设已激活)
    # 调用 V4 逻辑计算如果激活后的潜在反应度
    # 注意: V4 函数内部用到 residence_time，这里我们假设如果激活，反应时间为 (time - tau)
    # 但由于 V4 签名固定，且我们主要想复用其速率计算逻辑...
    # 实际上 V4 内部的 t_burn = residence - tau_ign.
    # 这里我们简化: 如果 gate 开启，则 alpha 按 V4 计算 (但时间很短)
    
    alpha_v4 = compute_miller_al_reaction_degree_v4(P, T, D0, residence_time, E_a_ign=E_a_ign)
    
    # 4. 应用门控
    alpha_final = alpha_v4 * gate
    
    # 额外强制: 如果 tau_ind >> current_time，则 alpha 必须极小
    # 物理上: CJ 面几乎完全由 TNT 驱动，铝是惰性的
    
    return alpha_final

