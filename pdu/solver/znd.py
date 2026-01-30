# pdu/solver/znd.py
import jax
import jax.numpy as jnp
from typing import Tuple, Any

from pdu.core.types import GasState, ParticleState, State
from pdu.thermo.implicit_eos import get_thermo_properties, get_sound_speed
from pdu.flux import compute_total_sources

"""
PDU V11.0 Tier 2: ZND 求解器 (GPU 优化版)
使用固定步长 RK4 积分，最大化 JIT 编译效率。
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
    atom_vec, coeffs_all, A_matrix, atomic_masses, eos_params = eos_data
    
    # 1. 状态解析 (隐式 EOS) - 代数求解，不作为微分变量
    P_pa, n_gas = get_thermo_properties(
        rho, T, 
        atom_vec, coeffs_all, A_matrix, atomic_masses, eos_params
    )
    
    # 获取精确声速
    a_gas = get_sound_speed(
        rho, T, n_gas, 
        atom_vec, coeffs_all, eos_params, atomic_masses
    )
    
    # 2. 化学动力学 (P-dependent)
    # Phase 1 修正: k_p 调整至合理量级
    # 在 30 GPa 下, dlam/dt ~ k_p * (30)^2 * (1-lam) = k_p * 900 * (1-lam)
    # 反应时间尺度 τ ~ 1/(k_p * 900) 秒
    # 若 τ ~ 1 μs = 1e-6 s, 则 k_p ~ 1e6/900 ~ 1e3
    p_gpa = P_pa / 1e9
    k_p = 1e3  # 修正后的反应速率常数 [1/(s·GPa^n)]
    n_p = 2.0
    # 物理约束: lam ∈ [0, 1], 反应只能前进，不能反向
    lam_clipped = jnp.clip(lam, 0.0, 1.0)
    remaining_reactant = jnp.maximum(1.0 - lam_clipped, 0.0)  # 确保非负
    dlam_dt = k_p * (jnp.maximum(p_gpa, 0.0)**n_p) * remaining_reactant
    
    # 3. 核心去奇异化变换
    M2 = (u / a_gas)**2
    singularity_factor = 1.0 - M2
    
    # 放热系数 sigma (简化版，后续用 AD 替代)
    gamma_eff = 3.0
    sigma = (gamma_eff - 1.0) * q_reaction / (a_gas**2 + 1e-10)
    
    # 分子项 (Numerator): 这是热力学驱动力
    numerator = rho * a_gas**2 * sigma * dlam_dt / (u + 1e-6)
    
    # 去奇异化后的导数
    dp_dxi = -numerator  # dP/dxi = -Numerator (分子项)
    dx_dxi = singularity_factor  # dx/dxi = 1 - M^2 (分母项)
    du_dxi = -dp_dxi / (rho * u + 1e-10)  # 动量守恒
    drho_dxi = -(rho / (u + 1e-6)) * du_dxi  # 质量守恒
    
    cv_gas = 2500.0
    dt_dxi = (-P_pa * du_dxi / (rho * u + 1e-10)) / cv_gas  # 能量守恒
    dlam_dxi = (dlam_dt / (u + 1e-6)) * singularity_factor
    
    return jnp.array([drho_dxi, du_dxi, dt_dxi, dlam_dxi, dx_dxi])

def solve_znd_profile(D, init_state, x_span, eos_data, drag_model, heat_model, q_reaction, n_steps=500):
    """
    Phase 1 简化版 ZND 求解器
    使用 5 维状态向量 [rho, u, T, lam, x]，移除 n 以消除刚性。
    """
    import diffrax
    
    # Phase 1: 简化状态向量 (5维)
    y0 = jnp.array([
        init_state.gas.rho, 
        init_state.gas.u, 
        init_state.gas.T, 
        init_state.gas.lam, 
        0.0  # x 初始位置
    ])
    args = (eos_data, drag_model, heat_model, D, q_reaction)
    
    # 使用显式求解器 Tsit5 (刚性已通过移除 n 消除)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)
    
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(znd_desingularized_field),
        solver,
        t0=0.0,
        t1=10.0,  # 伪时间预算 (增加以适应较慢的反应速率)
        dt0=1e-6,
        y0=y0,
        args=args,
        max_steps=10000,
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(steps=True),  # 保存完整轨迹用于调试
        adjoint=diffrax.RecursiveCheckpointAdjoint()
    )
    
    class Sol:
        def __init__(self, sol_diffrax):
            self.ts = sol_diffrax.ts
            # y 结构: [rho, u, T, lam, x]
            self.gas = GasState(
                rho=sol_diffrax.ys[:, 0],
                u=sol_diffrax.ys[:, 1],
                T=sol_diffrax.ys[:, 2],
                lam=sol_diffrax.ys[:, 3]
            )
            self.x = sol_diffrax.ys[:, 4]
            self.ys = self
            # 存储最终的导数用于 Loss 计算
            self._final_state = sol_diffrax.ys[-1]
            
    return Sol(sol)

def shooting_residual(D, init_state, x_span, eos_data, drag_model, heat_model, q_reaction):
    """
    V11 Phase 1 打靶残差函数
    
    专家建议的新 Loss 设计:
    L = ||Numerator||^2 + ||Denominator||^2 + w * (lam - 1)^2
    
    分子 (Numerator) = dP/dxi = 热力学驱动力
    分母 (Denominator) = dx/dxi = 1 - M^2 = 声速因子
    
    逻辑: 在 CJ 点，分子和分母必须同时趋零。
    """
    sol = solve_znd_profile(D, init_state, x_span, eos_data, drag_model, heat_model, q_reaction)
    
    # 终点状态
    u_f = sol.gas.u[-1]
    rho_f = sol.gas.rho[-1]
    T_f = sol.gas.T[-1]
    lam_f = sol.gas.lam[-1]
    
    atom_vec, coeffs_all, A_matrix, atomic_masses, eos_params = eos_data
    P_f, n_f = get_thermo_properties(rho_f, T_f, atom_vec, coeffs_all, A_matrix, atomic_masses, eos_params)
    cs_f = get_sound_speed(rho_f, T_f, n_f, atom_vec, coeffs_all, eos_params, atomic_masses)
    
    # 计算终点的分子项和分母项
    M_f = u_f / cs_f
    denominator = 1.0 - M_f**2  # dx/dxi 应趋于0
    
    # 重新计算分子项 (dP/dxi)
    p_gpa_f = P_f / 1e9
    k_p = 1e3  # 与向量场保持一致
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
