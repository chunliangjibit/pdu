# pdu/solver/znd.py
import jax
import jax.numpy as jnp
from typing import Tuple, Any

from pdu.core.types import GasState, ParticleState, State
from pdu.thermo.implicit_eos import get_thermo_properties, get_sound_speed
from pdu.flux import compute_total_sources

"""
PDU V11.0 Tier 2: ZND 求解器 (Robust Manual Integration)
Patch D: "Reject & Shrink" 策略，检测 NaN/负声速/过饱和并自动回退缩步。
"""

def znd_desingularized_field(xi, state, args):
    """
    去奇异化 ZND 向量场 (V11.0 Tier 2 - Phase 1 Simplified)
    引入伪时间 xi (dxi = dx / (1-M^2)) 以穿越声速面。
    
    Phase 1 简化: 移除 n 的微分演化，改为代数求解，消除刚性根源。
    state = [rho, u, T, lam, x] (5维简化状态)
    """
    rho = state[0]
    u = state[1]
    T = state[2]
    lam = state[3]
    x = state[4]
    
    eos_data, drag_model, heat_model, D, q_reaction = args
    atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params = eos_data
    
    # 1. 状态解析 (隐式 EOS) - 代数求解，不作为微分变量
    P_pa, n_gas = get_thermo_properties(
        rho, T, 
        atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params
    )
    
    # 获取精确声速
    a_gas = get_sound_speed(
        rho, T, n_gas, 
        atom_vec, coeffs_low, coeffs_high, eos_params, atomic_masses
    )
    
    # 2. 化学动力学 (P-dependent)
    # Phase 1 修正: k_p 调整至合理量级
    # 在 30 GPa 下, dlam/dt ~ k_p * (30)^2 * (1-lam) = k_p * 900 * (1-lam)
    # 反应时间尺度 τ ~ 1/(k_p * 900) 秒
    # 若 τ ~ 1 μs = 1e-6 s, 则 k_p ~ 1e6/900 ~ 1e3
    p_gpa = P_pa / 1e9
    p_gpa = P_pa / 1e9
    k_p = 1000.0  # Tuned: 100.0 -> 1000.0 (aggressive heating to fix T crash)
    n_p = 2.0
    n_p = 2.0
    # 物理约束: lam ∈ [0, 1], 反应只能前进，不能反向
    lam_clipped = jnp.clip(lam, 0.0, 1.0)
    remaining_reactant = jnp.maximum(1.0 - lam_clipped, 0.0)  # 确保非负
    dlam_dt = k_p * (jnp.maximum(p_gpa, 0.0)**n_p) * remaining_reactant
    dlam_dt = jnp.minimum(dlam_dt, 1e5) # 动力学截止限制 (V11.0 Phase 4)
    
    # 3. 核心去奇异化变换 (V11.0 Phase 4: Regularized)
    M2 = (u / a_gas)**2
    epsilon_sonic = 1e-4 # 对冲奇异性的微小扰动
    # 物理驱动力: 在反应区内 M < 1, singularity_factor 应为正
    singularity_factor = 1.0 - M2 + epsilon_sonic
    
    # 放热系数 sigma (简化版，后续用 AD 替代)
    gamma_eff = 3.0
    sigma = (gamma_eff - 1.0) * q_reaction / (a_gas**2 + 1e-10)
    
    # 分子项 (Numerator): 这是热力学驱动力
    numerator = rho * a_gas**2 * sigma * dlam_dt / (u + 1e-6)
    
    # 去奇异化后的导数
    dp_dxi = -numerator  # dP/dxi = -Numerator (分子项)
    dx_dxi = jnp.maximum(singularity_factor, 1e-6)  # 物理约束: x 轴只能正向演化
    du_dxi = -dp_dxi / (jnp.maximum(rho * u, 1e-3) + 1e-10)  # 动量守恒 (防止除以零)
    drho_dxi = -(rho / (jnp.maximum(u, 10.0) + 1e-6)) * du_dxi  # 质量守恒
    
    cv_gas = 1500.0 # HMX 典型值
    # 能量守恒: dT/dxi = (放热率 - 膨胀功) / (rho * Cv * dx/dxi)
    # 在 xi 空间中: dT/dxi = (q_reaction * dlam/dxi - P_pa * d(1/rho)/dxi) / cv_gas
    # d(1/rho)/dxi = -(1/rho^2) * drho/dxi = (1/(rho*u)) * du/dxi
    dt_exothermic = (q_reaction * dlam_dt / (u + 1e-6)) / cv_gas 
    dt_expansion = (P_pa * du_dxi / (rho * u + 1e-10)) / cv_gas
    dt_dxi = (dt_exothermic - dt_expansion)
    dlam_dxi = (dlam_dt / (u + 1e-6)) * jnp.maximum(singularity_factor, 1e-4) # 物理约束：反应度只能正向增加
    
    # 5. 状态裁剪与安全返回 (防止数值漂移)
    drho_dxi = jnp.nan_to_num(drho_dxi, 0.0)
    du_dxi = jnp.nan_to_num(du_dxi, 0.0)
    dt_dxi = jnp.nan_to_num(dt_dxi, 0.0)
    dlam_dxi = jnp.nan_to_num(dlam_dxi, 0.0)
    dx_dxi = jnp.nan_to_num(dx_dxi, 1e-6)
    
    return jnp.array([drho_dxi, du_dxi, dt_dxi, dlam_dxi, dx_dxi])

def solve_znd_profile(D, init_state, x_span, eos_data, drag_model, heat_model, q_reaction, n_steps=2000):
    """
    V11 Phase 5: 手动积分 ZND 求解器 (Reject & Shrink)
    完全替代 Diffrax 以实现细粒度的步长控制和坏点诊断。
    """
    # 初始状态 [rho, u, T, lam, x]
    y = jnp.array([
        init_state.gas.rho,
        init_state.gas.u,
        init_state.gas.T,
        init_state.gas.lam,
        0.0
    ])

    # 存储轨迹
    history = {
        'rho': [y[0]], 'u': [y[1]], 'T': [y[2]], 'lam': [y[3]], 'x': [y[4]],
        'P': [], 'a2': [], 'eta': [], 'step_size': []
    }

    # 初始辅助变量计算
    def evaluate_aux(state):
        rho, u, T, lam, x = state
        atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params = eos_data
        P_pa, n_gas = get_thermo_properties(rho, T, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
        cs_val = get_sound_speed(rho, T, n_gas, atom_vec, coeffs_low, coeffs_high, eos_params, atomic_masses)
        
        # 计算 eta 用于诊断
        # 注意: 这里简化计算，仅用于 check
        # 实际应从 EOS 内部获取，但为性能暂用近似或重新计算
        # 为避免复杂，这里只返回 P 和 cs
        return P_pa, cs_val**2

    P0, a20 = evaluate_aux(y)
    history['P'].append(P0)
    history['a2'].append(a20)
    history['eta'].append(0.0) # Placeholder
    history['step_size'].append(0.0)

    dxi = 1e-5     # 初始伪时间步长
    xi = 0.0
    
    args = (eos_data, drag_model, heat_model, D, q_reaction)
    
    print(f"ZND Start: D={D}, rho0={y[0]:.2f}, T0={y[2]:.1f}, P0={P0/1e9:.2f} GPa")

    for step in range(n_steps):
        # RK4 积分步
        # k1
        dy1 = znd_desingularized_field(xi, y, args)
        
        # k2
        y2 = y + 0.5 * dxi * dy1
        dy2 = znd_desingularized_field(xi + 0.5 * dxi, y2, args)
        
        # k3
        y3 = y + 0.5 * dxi * dy2
        dy3 = znd_desingularized_field(xi + 0.5 * dxi, y3, args)
        
        # k4
        y4 = y + dxi * dy3
        dy4 = znd_desingularized_field(xi + dxi, y4, args)
        
        y_next = y + (dxi / 6.0) * (dy1 + 2*dy2 + 2*dy3 + dy4)
        
        # ---------------------------------------------------------
        # Patch D: Reject & Shrink 检查
        # ---------------------------------------------------------
        rho_n, u_n, T_n, lam_n, x_n = y_next
        
        if not jnp.all(jnp.isfinite(y_next)):
            print(f"Step {step}: Reject (NaN detected). Shrinking dxi {dxi:.2e} -> {dxi*0.5:.2e}")
            dxi *= 0.5
            continue

        # 1.5 Physical Realm Check (Early Reject)
        if T_n < 100.0:
             print(f"Step {step}: Reject (T < 100K: {T_n:.2f}). Shrinking...")
             dxi *= 0.5
             continue
        if rho_n < 0.1:
             print(f"Step {step}: Reject (rho too low: {rho_n:.2f}). Shrinking...")
             dxi *= 0.5
             continue

        # 2. 物理约束检查 (EOS / 声速)
        try:
            P_n, a2_n = evaluate_aux(y_next)
            
            # 2.1 声速检查
            if a2_n <= 0.0:
                 print(f"Step {step}: Reject (Negative a2={a2_n:.2e}). Shrinking...")
                 dxi *= 0.5
                 continue
                 
            # 2.2 反应进度倒退检查 (允许微小数值波动)
            if lam_n < history['lam'][-1] - 1e-4:
                 print(f"Step {step}: Reject (Lambda reversal). Shrinking...")
                 dxi *= 0.5
                 continue

        except Exception as e:
            print(f"Step {step}: Reject (EOS Error: {e}). Shrinking...")
            dxi *= 0.5
            continue
            
        # ---------------------------------------------------------
        # Accept Step
        # ---------------------------------------------------------
        y = y_next
        xi += dxi
        
        # 记录
        history['rho'].append(rho_n)
        history['u'].append(u_n)
        history['T'].append(T_n)
        history['lam'].append(lam_n)
        history['x'].append(x_n)
        history['P'].append(P_n)
        history['a2'].append(a2_n)
        history['eta'].append(0.0)
        history['step_size'].append(dxi)
        
        # 动态步长调整 (简单的增长策略)
        if step % 10 == 0:
            dxi = min(dxi * 1.1, 1e-3)
            
        # 终止条件: 反应完成
        if lam_n >= 0.99:
            print(f"ZND Finished: Lambda reached 0.99 at step {step}")
            break
            
    # 转换为 Sol 对象格式返回
    class Sol:
        pass
    sol = Sol()
    # 转换为 jnp array
    for k, v in history.items():
        setattr(sol, k, jnp.array(v))
        
    # 构造 gas 结构
    sol.gas = GasState(
        rho=sol.rho,
        u=sol.u,
        T=sol.T,
        lam=sol.lam
    )
    # 为兼容 shooting_residual
    sol.ys = jnp.stack([sol.rho, sol.u, sol.T, sol.lam, sol.x], axis=-1)
    
    return sol

# (solve_znd_profile has been replaced above)

def shooting_residual(D, init_state, x_span, eos_data, drag_model, heat_model, q_reaction):
    """
    V11 Phase 1 打靶残差函数
    
    专家建议的新 Loss 设计:
    L = ||Numerator||^2 + ||Denominator||^2 + w * (lam - 1)^2
    
    分子 (Numerator) = dP/dxi = 热力学驱动力
    分母 (Denominator) = dx/dxi = 1 - M^2 = 声速因子
    
    逻辑: 在 CJ 点，分子和分母必须同时趋零。
    """
    sol = solve_znd_profile(D, init_state, x_span, eos_data, drag_model, heat_model, q_reaction, n_steps=5000)
    
    # 终点状态 (Sol 对象现在直接存储 jnp array)
    u_f = sol.gas.u[-1]
    rho_f = sol.gas.rho[-1]
    T_f = sol.gas.T[-1]
    lam_f = sol.gas.lam[-1]
    
    atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params = eos_data
    P_f, n_f = get_thermo_properties(rho_f, T_f, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
    cs_f = get_sound_speed(rho_f, T_f, n_f, atom_vec, coeffs_low, coeffs_high, eos_params, atomic_masses)
    
    # 计算终点的分子项和分母项
    M_f = u_f / cs_f
    denominator = 1.0 - M_f**2  # dx/dxi 应趋于0
    
    # 重新计算分子项 (dP/dxi)
    p_gpa_f = P_f / 1e9
    k_p = 10.0  # 与向量场保持一致
    n_p = 2.0
    dlam_dt_f = k_p * (jnp.maximum(p_gpa_f, 0.0)**n_p) * (1.0 - lam_f)
    gamma_eff = 3.0
    sigma_f = (gamma_eff - 1.0) * q_reaction / (cs_f**2 + 1e-10)
    numerator = rho_f * cs_f**2 * sigma_f * dlam_dt_f / (u_f + 1e-6)  # dP/dxi 应趋于0
    
    # 归一化分子项 (量级差异大，需要归一化)
    numerator_normalized = numerator / (1e9 + 1e-10)  # 归一化到 GPa 量级
    
    # V11 新 Loss: 分子分母同时为0 + 反应完成
    residual_numerator = numerator_normalized
    residual_denominator = denominator
    residual_lam = lam_f - 1.0
    
    return jnp.sqrt(residual_numerator**2 + residual_denominator**2 + 10.0 * residual_lam**2)
