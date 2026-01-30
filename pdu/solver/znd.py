# pdu/solver/znd.py
import jax
import jax.numpy as jnp
from typing import Tuple, Any

from pdu.core.types import GasState, ParticleState, State
from pdu.thermo.implicit_eos import get_thermo_properties, get_sound_speed, _mixture_mass_and_mw_avg_kg_per_mol, _rho_to_kg_per_m3
from pdu.physics.eos import compute_total_helmholtz_energy, smooth_floor
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

# =============================================================================
# Patch E (Expert V1): Thermodynamic Projection Solver
# =============================================================================

class NewtonConfig:
    max_it: int = 20
    tol: float = 1e-3 # Pa (Absolute tolerance). 1e-3 Pa is extremely tight for 50 GPa.
    V_min: float = 1e-8  # m3/kg (approx)
    T_min: float = 100.0 # K
    max_ls: int = 5

def thermo_from_helmholtz(n, V_total, T, coeffs_low, coeffs_high,
                          eps_vec, r_star_vec, alpha_vec, lambda_vec,
                          solid_mask, solid_v0,
                          n_fixed_solids=0.0, v0_fixed_solids=10.0, e_fixed_solids=0.0,
                          r_star_rho_corr=0.0,
                          mw_avg=1.0): # mw_avg is placeholder if we calculate mass internally? 
    # Expert code suggestion: "mw_avg=mw_avg".
    # We must ensure consistent mass usage.
    
    # Calculate A and derivatives
    # Note: compute_total_helmholtz_energy returns A value. 
    # We need derivatives.
    
    # We need to wrap it to get derivatives w.r.t V_total and T
    def A_of_VT(Vc, Tc):
        return compute_total_helmholtz_energy(
            n, Vc, Tc, 
            coeffs_low, coeffs_high,
            eps_vec, r_star_vec, alpha_vec, lambda_vec,
            solid_mask, solid_v0,
            n_fixed_solids, v0_fixed_solids, e_fixed_solids,
            r_star_rho_corr,
            mw_avg=mw_avg
        )

    # Derivatives
    # compute_total_helmholtz_energy is jitted, so we can grad it.
    # But it's better to use the same logic as implicit_eos.py/get_sound_speed to capture gradients cleanly.
    
    # Use jax.value_and_grad?
    # Or just grad.
    
    A_val = A_of_VT(V_total, T)
    
    dA_dV = jax.grad(A_of_VT, argnums=0)
    dA_dT = jax.grad(A_of_VT, argnums=1)
    
    d2A_dV2 = jax.grad(lambda v, t: dA_dV(v, t), argnums=0)
    d2A_dVdT = jax.grad(lambda v, t: dA_dV(v, t), argnums=1)
    d2A_dT2 = jax.grad(lambda v, t: dA_dT(v, t), argnums=1)
    
    # Compute values
    A_V = dA_dV(V_total, T)
    A_T = dA_dT(V_total, T)
    A_VV = d2A_dV2(V_total, T)
    A_VT = d2A_dVdT(V_total, T)
    A_TT = d2A_dT2(V_total, T)

    P = -A_V
    P_T = -A_VT
    P_V = -A_VV
    
    U_total = A_val - T * A_T
    Cv_total = -T * A_TT
    
    return P, U_total, Cv_total, P_T, P_V

def project_state_to_conservation(n, V_guess, T_guess,
                                  J, Pi, E,
                                  eos_data, newton_cfg=NewtonConfig()):
    
    atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params = eos_data
    
    # Consistent Mass Logic
    m_kg, mw_avg_kg_per_mol, n_tot = _mixture_mass_and_mw_avg_kg_per_mol(n, atom_vec, atomic_masses)
    
    # Prepare EOS args (unpack eos_params)
    # eos_params is list/tuple: [eps_vec, r_star_vec, alpha_vec, lambda_vec, solid_mask, solid_v0, n_fixed, v0_fixed, e_fixed, r_star_rho_corr]
    # Check implicit_eos.py usage
    eps_vec, r_star_vec, alpha_vec, lambda_vec, solid_mask, solid_v0, n_fixed, v0_fixed, e_fixed, r_star_rho_corr = eos_params

    # Solve R1(V,T)=0, R2(V,T)=0
    # Inputs V_guess, T_guess.
    
    # If V_guess is m3? or m3/kg?
    # Expert code says: "V = exp(yV)", "rho = m_kg / V". So V is Volume of system (m3).
    # ensure guess is valid magnitude
    V_safe = jnp.maximum(V_guess, 1e-8)
    T_safe = jnp.maximum(T_guess, 100.0)
    
    yV = jnp.log(V_safe)
    yT = jnp.log(T_safe)
    
    def loop_body(carry):
        yV, yT, it, converged = carry
        
        V = jnp.exp(yV)
        T = jnp.exp(yT)
        
        # Call EOS
        # V must be passed in unit expected by compute_total_helmholtz_energy (m3? cm3?)
        # expert said: "compute_total looks for V_total". implicit_eos passes V_total_m3.
        # But wait, implicit_eos passes V_total_m3?
        # Let's check implicit_eos.py again.
        # "V_cm3 = V_total_m3 * 1e6"
        # "return compute_total...(..., Vc=V_cm3, ...)" in `A_of_VT` wrapper.
        # So compute_total expects cm3.
        
        V_cm3 = V * 1e6
        
        P_pa, U_total_j, Cv_total_j_k, P_T_pa_k, P_V_pa_m3 = thermo_from_helmholtz(
            n, V_cm3, T, 
            coeffs_low, coeffs_high,
            eps_vec, r_star_vec, alpha_vec, lambda_vec,
            solid_mask, solid_v0,
            n_fixed, v0_fixed, e_fixed,
            r_star_rho_corr,
            mw_avg=mw_avg_kg_per_mol
        )
        
        # Convert Derivatives from cm3 basis if needed?
        # thermo_from_helmholtz gradients are w.r.t V_cm3.
        # P = -dA/dV_cm3 * 1e6?
        # thermo_from_helmholtz should likely return SI units to be clean.
        # Let's verify thermo_from_helmholtz implementation above.
        # It calls A_of_VT(V_total, T). If V_total is cm3, then A_V is J/cm3.
        # P = -A_V (J/cm3 = MPa). P_pa = P * 1e6.
        
        # Let's fix thermo_from_helmholtz to return SI P, P_T, P_V.
        # If input V is cm3:
        P_val = P_pa * 1e6
        P_T_val = P_T_pa_k * 1e6
        P_V_val = P_V_pa_m3 * 1e12 # J/cm6 -> Pa/m3?
        # A_VV is J/cm6. P_V = -A_VV. Unit: J/cm6.
        # J/cm6 = (N m) / (1e-2 m)^6 = 1e12 N m / m^6 = 1e12 N/m5?
        # P is N/m2. P_V is Pa / m3 = N/m5.
        # So factor 1e12 from cm6 to m6 is correct (1/1e-12 = 1e12).
        # WAIT. cm6 = 1e-12 m6.
        # J/cm6 = J / 1e-12 m6 = 1e12 J/m6.
        # P_V should be Pa/m3? No. P is in J/V?
        # dP/dV. P=J/cm3. dP/dV = J/cm6.
        # 1 J/cm6 = 1e12 J/m6 = 1e12 (N/m2) / m3 ? No. 1 J/m3 = 1 Pa.
        # dP/dV in SI = Pa / m3.
        # J/m6 / 1 = Pa/m3.
        # So yes, 1e12 factor.
        
        # Redefine returned values of thermo_from_helmholtz to be raw (J/cm3 basis) or SI?
        # Let's adjust loop logic to handle SI conversion explicitly.
        
        P = P_pa * 1e6 # Pa
        e = U_total_j / m_kg # J/kg
        rho = m_kg / V
        
        R1 = P + (J**2)*V/m_kg - Pi
        R2 = e + P/rho + 0.5*(J/rho)**2 - E
        
        # Jacobian SI
        P_T = P_T_pa_k * 1e6
        P_V = P_V_pa_m3 * 1e12
        cv_mass = Cv_total_j_k / m_kg
        
        dR1_dV = P_V + (J**2)/m_kg
        dR1_dT = P_T

        # Exp code: dR2_dV = (T*P_T + V*P_V)/m_kg + (J*J)*V/(m_kg*m_kg)
        # Note: P cancellation trick relies on U_V = -P + T P_T.
        # U_total_V_cm3 = A_V - T A_VT = -P_raw + T (-P_T_raw).
        # U_total_V_m3 = (-P/1e6 + T * (-P_T/1e6)) * 1e6 * 1e6 ?? 
        # Easier to trust SI formulation.
        
        dR2_dV = (T * P_T + V * P_V) / m_kg + (J**2 * V) / (m_kg**2)
        dR2_dT = cv_mass + P_T/rho
        
        J11 = dR1_dV * V
        J12 = dR1_dT * T
        J21 = dR2_dV * V
        J22 = dR2_dT * T
        
        det = J11*J22 - J12*J21
        dyV = (J22*(-R1) - J12*(-R2)) / (det + 1e-20)
        dyT = (-J21*(-R1) + J11*(-R2)) / (det + 1e-20)
        
        # Damping
        damping = 1.0
        # Simple step limit
        dist = jnp.sqrt(dyV**2 + dyT**2)
        scale = jnp.minimum(1.0, 2.0 / (dist + 1e-10))
        
        yV_new = yV + dyV * scale
        yT_new = yT + dyT * scale
        
        err = jnp.sqrt(R1**2 + R2**2)
        converged = err < newton_cfg.tol

        # Debug Newton
        # jax.debug.print("Newton It {}: R1={:.2e}, R2={:.2e}, det={:.2e}, dyV={:.2e}, dyT={:.2e}, T={:.1f}", it, R1, R2, det, dyV, dyT, T)
        
        return (yV_new, yT_new, it+1, converged)

    # Jax While Loop
    val = (yV, yT, 0, False)
    cond = lambda v: (v[2] < newton_cfg.max_it) & (~v[3])
    final_val = jax.lax.while_loop(cond, loop_body, val)
    
    yV_f, yT_f, it_f, conv_f = final_val
    V_f = jnp.exp(yV_f)
    T_f = jnp.exp(yT_f)
    rho_f = m_kg / V_f
    
    # Return solution and convergence flag
    return V_f, T_f, rho_f, conv_f

def solve_znd_projection(D, init_state, x_span, eos_data, drag_model, heat_model, q_reaction, n_steps=500):
    """
    V11 Patch E: Chemical Explicit + Thermodynamic Projection
    Replaces explicit integration with algebraic projection on the Rayleigh-Hugoniot path.
    """
    atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params = eos_data
    
    # 1. Constants & Initialization
    rho_vn = init_state.gas.rho
    u_vn = init_state.gas.u
    T_vn = init_state.gas.T
    
    # Get accurate VN properties
    P_vn_pa, n_vn = get_thermo_properties(rho_vn, T_vn, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
    m_kg, _, _ = _mixture_mass_and_mw_avg_kg_per_mol(n_vn, atom_vec, atomic_masses)
    V_vn = m_kg / (rho_vn * 1000.0) # rho is g/cm3 -> kg/m3 conversion needed for SI Volume
    
    # Calculate System Constants (Unreacted/VN State)
    # The flow is steady, so Mass, Momentum, Energy fluxes are constant.
    # CONSTANTS are based on the INITIAL UNREACTED state (upstream of shock).
    # But init_state passed here is usually VN (Post-Shock).
    # Momentum J and Pi are preserved across shock.
    # Energy E is preserved.
    # J = rho * u. rho in SI (kg/m3).
    # init_state.rho is g/cm3.
    J = (rho_vn * 1000.0) * u_vn
    Pi = P_vn_pa + J**2 / (rho_vn * 1000.0)
    
    # Calculate Total Energy E from VN state
    # Assumption: The VN state provided matches the Rayleight line of the detonation D.
    # H_vn + 0.5 u_vn^2 = E_total
    # Note: We treat the fluid as "Product Mixture" conceptually + Potential Heat.
    # Since n_vn comes from get_thermo_properties, it is the Equilibrium Composition at VN (Shocked).
    # This is our reference "Product" EOS state.
    # But physically, VN is Unreacted. 
    # The "Heat Model" (q_reaction) implies we have extra potential energy (1-lam)*Q.
    # So E_total (Conserved) = H_prod(VN) + 0.5 u_vn^2 + (1 - lam_vn)*Q
    # Typically lam_vn ~ 0.
    lam_0 = 0.0
    
    eps_vec, r_star_vec, alpha_vec, lambda_vec, solid_mask, solid_v0, n_fixed, v0_fixed, e_fixed, r_star_rho_corr = eos_params
    mw_avg_vn = m_kg / jnp.sum(n_vn)
    
    # Get VN Internal Energy (Product Basis)
    V_vn_cm3 = V_vn * 1e6
    _, U_vn_total, _, _, _ = thermo_from_helmholtz(
        n_vn, V_vn_cm3, T_vn, 
        coeffs_low, coeffs_high,
        eps_vec, r_star_vec, alpha_vec, lambda_vec,
        solid_mask, solid_v0,
        n_fixed, v0_fixed, e_fixed,
        r_star_rho_corr,
        mw_avg=mw_avg_vn
    )
    e_vn = U_vn_total / m_kg # J/kg
    
    
    E_total = e_vn + P_vn_pa/(rho_vn * 1000.0) + 0.5 * u_vn**2 + (1.0 - lam_0) * q_reaction
    
    print(f"ZND Projection Start: D={D}, P_vn={P_vn_pa/1e9:.2f} GPa, T_vn={T_vn:.0f} K, E_tot={E_total:.2e}")

    # 2. Stepping Loop
    # We step lambda from 0 to 1
    lam_seq = jnp.linspace(0.0, 0.99, n_steps) # Avoid 1.0 singularity
    
    history = {
        'rho': [], 'u': [], 'T': [], 'lam': [], 'x': [], 
        'P': [], 'a': [], 'M': [], 'sw_det': []
    }
    
    # Init vars
    V_curr = V_vn
    T_curr = T_vn
    n_curr = n_vn
    x_curr = 0.0
    t_curr = 0.0
    
    # Store step 0
    a_vn_val = get_sound_speed(rho_vn, T_vn, n_vn, atom_vec, coeffs_low, coeffs_high, eos_params, atomic_masses)
    history['rho'].append(rho_vn)
    history['u'].append(u_vn)
    history['T'].append(T_vn)
    history['lam'].append(lam_0)
    history['x'].append(0.0)
    history['P'].append(P_vn_pa)
    history['a'].append(a_vn_val)
    history['M'].append(u_vn/a_vn_val)
    history['sw_det'].append(1.0) # Dummy det

    for i in range(1, n_steps):
        lam = lam_seq[i]
        dlam = lam - lam_seq[i-1]
        
        # A. Target Energy for Product EOS
        # E_prod_target = E_total - (1-lam)*Q
        E_prod_target = E_total - (1.0 - lam) * q_reaction
        
        # B. Composition Guess
        # Use previous n_curr as guess for Projection
        # (Assuming composition changes slowly or is frozen for the step)
        
        # C. Project (V, T)
        # Solve R1, R2 for V, T
        V_new, T_new, rho_new, conv = project_state_to_conservation(
            n_curr, V_curr, T_curr,
            J, Pi, E_prod_target,
            eos_data
        )
        
        if not conv:
            print(f"Step {i} (lam={lam:.3f}): Projection Failed. Stopping.")
            break
            
        # D. Update Composition (Equilibrium at new V, T)
        # PDU assumption: Gas is always in local equilibrium (or partial equilibrium)
        P_tmp, n_new = get_thermo_properties(rho_new, T_new, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params, n_init=n_curr)
        
        # Optional: Re-Project with new n? (Corrector)
        # For precision, yes.
        V_final, T_final, rho_final, conv_final = project_state_to_conservation(
            n_new, V_new, T_new,
            J, Pi, E_prod_target,
            eos_data
        )
        
        if not conv_final:
             print(f"Step {i}: Corrector Failed.")
             break
        
        # Update state
        V_curr = V_final
        T_curr = T_final
        n_curr = n_new
        rho_curr = rho_final
        u_curr = J / rho_curr
        P_curr = Pi - J**2 / rho_curr # Consistent P from Momentum
        
        # E. Integrate Space/Time
        # Rate Calculation
        P_gpa = P_curr / 1e9
        k_p = 1000.0 # Consistent with older code
        n_p = 2.0
        rate = k_p * (jnp.maximum(P_gpa, 0.0)**n_p) * (1.0 - lam)
        rate = jnp.maximum(rate, 1e-10) # Avoid div zero
        
        dt = dlam / rate
        dx = u_curr * dt
        
        x_curr += dx
        t_curr += dt
        
        # F. Sonic Check (Patch E3)
        # Verify if we hit CJ
        a_curr = get_sound_speed(rho_curr, T_curr, n_curr, atom_vec, coeffs_low, coeffs_high, eos_params, atomic_masses)
        M_curr = u_curr / a_curr
        
        sonic_crit = 1.0 - M_curr
        
        # Store
        history['rho'].append(rho_curr)
        history['u'].append(u_curr)
        history['T'].append(T_curr)
        history['lam'].append(lam)
        history['x'].append(x_curr)
        history['P'].append(P_curr)
        history['a'].append(a_curr)
        history['M'].append(M_curr)
        history['sw_det'].append(0.0) # placeholder

        if jnp.abs(sonic_crit) < 1e-3:
            print(f"Sonic Point Reached at lam={lam:.3f}, M={M_curr:.4f}. Stopping.")
            break
        
        if M_curr > 1.02: # Overshoot check (if we jumped to Weak solution?)
            # Usually Strong solution starts M < 1 and goes to M=1.
            # If M > 1, we might have crossed CJ.
            print(f"Supersonic flow detected (M={M_curr:.3f}). Stopping.")
            break

    # Pack Solution
    class Sol:
        pass
    sol = Sol()
    # Convert to jnp arrays
    for k, v in history.items():
        setattr(sol, k, jnp.array(v))
        
    sol.gas = GasState(
        rho=sol.rho,
        u=sol.u,
        T=sol.T,
        lam=sol.lam
    )
    # Fake ys for compatibility if need be
    return sol

# Alias to replace old solver
solve_znd_profile = solve_znd_projection

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
