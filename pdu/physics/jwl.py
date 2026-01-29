
"""
JWL 状态方程拟合模块

提供 JWL (Jones-Wilkins-Lee) 状态方程的拟合功能。
P = A(1 - omega/R1*V)exp(-R1*V) + B(1 - omega/R2*V)exp(-R2*V) + omega*E/V
其中 V = v/v0 (相对比容)
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Tuple

@dataclass
class JWLParams:
    A: float # GPa
    B: float # GPa
    R1: float
    R2: float
    omega: float
    E0: float # GPa (Energy density per unit volume initial)
    fit_mse: float = 0.0 # 对数域拟合均方误差

def jwl_pressure(V_rel, A, B, R1, R2, omega, E_per_vol):
    """JWL Pressure Equation (Principal Isentrope assumption E(V))
    
    Usually for fitting, we fit P(V) on the isentrope.
    On the isentrope, E is a function of V.
    P_s(V) = A exp(-R1 V) + B exp(-R2 V) + C V^(-1 - omega)
    
    Wait, the standard analytic isentrope form of JWL is:
    P_s(V) = A exp(-R1 V) + B exp(-R2 V) + C * V^(-(1+omega))
    
    The standard EOS form is:
    P(V,E) = A(1 - w/R1V)e^(-R1V) + B(1 - w/R2V)e^(-R2V) + wE/V
    
    If we fit the isentrope P-V data, we should use the P_s(V) form.
    Then we ensure thermodynamic consistency C = ...? 
    Usually C relates to initial energy.
    
    Args:
        V_rel: v/v0
    """
    # Using P_s form for stable fitting
    # P = A * exp(-R1 * V) + B * exp(-R2 * V) + K / V**(omega + 1)
    
    # But user wants A, B, R1, R2, omega. 
    # Usually we fit this P_s form.
    pass

def fit_jwl_from_isentrope(
    V_rel_array, P_array, rho0, E0, D_cj, P_cj_theory, 
    exp_priors=None,
    constraint_P_cj: float = None,  # GPa
    constraint_D_exp: float = None, # m/s (for Rayleigh line)
    constraint_total_energy: float = None, # GPa (Total Available Energy E0_reactive)
    method: str = 'Nelder-Mead' # 'Nelder-Mead' or 'PSO' (V10 Upgrade)
):
    """
    V8.7: Prior-Aware Robust Fitting with Engineering Bias & Total Energy Constraint
    
    Args:
        constraint_P_cj: Optional experimental P_CJ to anchor the fitting.
        constraint_total_energy: Optional Total Energy (GPa) to constrain the integral area under JWL.
                                 Used for "Two-Step" fitting (Inert P_CJ but Reactive Energy).
    """
    from scipy.optimize import minimize
    import numpy as np

    # 1. 物理锚点数据 (CJ 点)
    V_cj_rel = V_rel_array[0]
    P_cj = P_array[0]
    
    # 2. 计算物理目标 Gamma (绝热指数)
    if P_cj_theory < 1e-6:
        gamma_cj_target = 3.0 # Fallback for failed physics
        print(f"WARNING: P_cj_theory is zero or invalid ({P_cj_theory}). Using fallback Gamma=3.0")
    else:
        gamma_cj_target = (rho0 * (D_cj**2) * 1e-6) / P_cj_theory - 1.0 
    
    # [V8.6] Engineering Constraint Target
    target_P_cj = None
    target_V_cj = None
    if constraint_P_cj is not None:
        target_P_cj = constraint_P_cj
        if constraint_D_exp is not None:
            # Calculate consistent V_cj on the Rayleigh line
            term = (constraint_P_cj * 1e9) / (rho0 * (constraint_D_exp**2))
            target_V_cj = 1.0 - term
            print(f"[JWL Bias] Enforcing P_CJ={target_P_cj} GPa, V_CJ={target_V_cj:.4f} (D={constraint_D_exp})")
        else:
            # Use geometric intersection or just V_cj_rel from inert calc?
            # If we don't have D_exp, we can't fully define Rayleigh.
            # Fallback: Assume V_cj_rel from prediction is correct-ish? No, if P changed, V changes.
            # For Two-Step (V8.7), we usually trust the Inert V_cj, just anchor P to Inert P_cj (which is P_cj matches).
            # So if D_exp is None, we assume target_V_cj = V_cj_rel
            target_V_cj = V_cj_rel
    
    # 3. 对数域数据
    V_rel_array = jnp.array(V_rel_array)
    P_array = jnp.array(P_array)
    y_log = jnp.log(jnp.maximum(P_array, 1e-4))
    
    # 4. 目标函数 (JAX-compatible for PSO JIT)
    def objective(params):
        A, B, R1, R2, w, C = params
        
        # ========== V10.6 增强物理约束 (专家意见响应) ==========
        # Target range: [0.25, 0.45] for CHNO
        bad = (A < 50.0) | (C < 0.0) | (w < 0.25) | (w > 0.45) | (R1 < (R2 + 1.0)) | (R2 < 0.3)
        penalty_barrier = jnp.where(bad, 1e9, 0.0)
        
        # [V10.6] Topology Constraint (A vs B)
        ratio_BA = B / (A + 1e-3)
        penalty_topology = jnp.where(ratio_BA > 0.1, (ratio_BA - 0.1)**2 * 10000.0, 0.0)

        # [V10.5 关键改进] 声速稳定性检查 (Stability Watchdog)
        # c² = -V² * dP/dV
        # 我们需要在膨胀全历程 (V=1 到 V=10) 确保 c² > 0
        def check_sound_speed(V):
            # 等熵线导数 dP/dV
            # P = A exp(-R1 V) + B exp(-R2 V) + C V^-(1+w)
            term1 = -A * R1 * jnp.exp(-R1 * V)
            term2 = -B * R2 * jnp.exp(-R2 * V)
            term3 = -C * (1.0 + w) * (V**-(2.0 + w))
            dP_dV = term1 + term2 + term3
            c_squared = -V**2 * dP_dV
            return c_squared
        
        # 采样点：覆盖近场到远场
        V_check_points = jnp.array([1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0])
        c2_values = jax.vmap(check_sound_speed)(V_check_points)
        
        # 只要有一个点 c2 <= 0，就说明出现了物理塌陷（如压力随体积增加而上升）
        # 惩罚项必须足够大，以强制 PSO 避开这些区域
        stability_penalty = jnp.sum(jnp.where(c2_values <= 1e-4, 1e10, 0.0))
        
        # (G) 曲率约束: d²P/dV² > 0 (凸性)
        def check_curvature(V):
            d2P_dV2 = A * R1**2 * jnp.exp(-R1 * V) + \
                      B * R2**2 * jnp.exp(-R2 * V) + \
                      C * (1.0 + w) * (2.0 + w) * (V**-(3.0 + w))
            return d2P_dV2
        
        curvature_values = jax.vmap(check_curvature)(V_check_points)
        curvature_penalty = jnp.sum(jnp.where(curvature_values < 0, 1e7, 0.0))
        
        # ========== 原有 Loss 逻辑 ==========
        # 模型预测 (Isentrope 形式)
        P_pred = A * jnp.exp(-R1 * V_rel_array) + \
                 B * jnp.exp(-R2 * V_rel_array) + \
                 C / (jnp.maximum(V_rel_array, 1e-4)**(1.0 + w))
        
        y_pred_log = jnp.log(jnp.maximum(P_pred, 1e-6))
        
        # (A) Log-MSE Loss
        weights = jnp.where(V_rel_array < 1.2, 2.0,
                           jnp.where(V_rel_array < 3.0, 1.5, 1.0))
        loss_fit = jnp.mean(weights * (y_pred_log - y_log)**2) * 5000.0 # High weight for V10.6
        
        # (B) P_CJ Anchor Penalty (V10.6 Harder)
        P_fit_cj = A * jnp.exp(-R1 * V_cj_rel) + \
                   B * jnp.exp(-R2 * V_cj_rel) + \
                   C / (jnp.maximum(V_cj_rel, 1e-4)**(1.0 + w))
        
        rel_err = jnp.abs(P_fit_cj - P_cj) / P_cj
        loss_anchor_P = jnp.where(rel_err > 0.01, (rel_err - 0.01)**2 * 200000.0, 0.0)

        # (D) Total Energy Constraint
        V_start = target_V_cj if target_V_cj is not None else V_cj_rel
        term1_e = (A / R1) * jnp.exp(-R1 * V_start)
        term2_e = (B / R2) * jnp.exp(-R2 * V_start)
        term3_e = (C / jnp.maximum(w, 1e-3)) * (V_start**(-w))
        E_integral = term1_e + term2_e + term3_e
        loss_energy = ((E_integral - target_E) / target_E) ** 2 * 10000.0
            
        return (loss_fit + loss_anchor_P + loss_energy + 
                penalty_barrier + stability_penalty + curvature_penalty + penalty_topology)

    # 5. 优化准备 (使用先验作为初始值)
    if exp_priors:
        ep = exp_priors
        x0 = [ep['A'], ep['B'], ep['R1'], ep['R2'], ep['omega'], 1.0]
    
    # V10.4 P0-3: CJ Anchor Method (Dimensionality Reduction)
    if method == 'CJ_ANCHOR':
        from pdu.calibration.pso import PSOOptimizer
        print(f"[V10.4 Anchor] Running CJ-Anchored Optimization (3D Search)...")
        
        # Calculate Targets
        target_E = constraint_total_energy if constraint_total_energy is not None else (E0 * 1000.0)
        
        # Calculate Slope at CJ from Gamma
        # dP/dV = -Gamma * P / V
        slope_cj = -gamma_cj_target * P_cj / V_cj_rel
        
        def anchor_obj(p):
            # p: [R1, R2, w]
            R1, R2, w = p
            
            # 1. Solve Linear System
            A, B, C = solve_jwl_linear_params(R1, R2, w, V_cj_rel, P_cj, slope_cj, target_E)
            
            # 2. Hard Physical Constraints (Vectorized)
            # Check for NaN (Singular matrix)
            is_nan = jnp.isnan(A) | jnp.isnan(B) | jnp.isnan(C)
            
            # Non-negativity
            penalty_neg = jnp.where(A < 0, 1e8 + jnp.abs(A)*100, 0.0) + \
                          jnp.where(B < 0, 1e8 + jnp.abs(B)*100, 0.0) + \
                          jnp.where(C < 0, 1e8 + jnp.abs(C)*100, 0.0)
            
            # R1 > R2 ordering
            penalty_order = jnp.where(R1 < R2 + 0.1, 1e9, 0.0)
            
            # Combine penalties
            penalty_constraints = jnp.where(is_nan, 1e10, penalty_neg + penalty_order)
            
            # Logic: If constraints violated, return penalty. Else compute MSE.
            # But we must compute MSE anyway to keep shapes static, or use where.
            
            params_full = jnp.array([A, B, R1, R2, w, C])
            val_fit = objective(params_full)
            
            return jnp.where(penalty_constraints > 0, penalty_constraints, val_fit)
            
        # 3D Search Space: R1, R2, w
        lb = jnp.array([1.5, 0.5, 0.1])
        ub = jnp.array([12.0, 4.0, 1.0])
        
        pso = PSOOptimizer(anchor_obj, lb, ub, num_particles=50) # Faster
        best_x_pso, best_f_pso = pso.optimize(num_iterations=100)
        
        R1, R2, w = best_x_pso
        A, B, C = solve_jwl_linear_params(R1, R2, w, V_cj_rel, P_cj, slope_cj, target_E)
        
        # Final refinement? No, analytical is exact.
        # Just return results.
        print(f"[V10.4 Anchor] Result: A={A:.1f}, B={B:.1f}, w={w:.3f}")
        
        # Calculate final MSE
        P_final = A * np.exp(-R1 * V_rel_array) + \
                 B * np.exp(-R2 * V_rel_array) + \
                 C / (V_rel_array**(1.0 + w))
        final_mse = np.mean((np.log(np.maximum(P_final, 1e-6)) - y_log)**2)
        
        return JWLParams(A=float(A), B=float(B), R1=float(R1), R2=float(R2), omega=float(w), E0=float(E0), fit_mse=float(final_mse))

        return JWLParams(A=float(A), B=float(B), R1=float(R1), R2=float(R2), omega=float(w), E0=float(E0), fit_mse=float(final_mse))

    # V10.6: Seeded PSO search
    if method == 'RELAXED_PENALTY':
        from pdu.calibration.pso import PSOOptimizer
        print(f"[V10.6 Constrained] Running Seeded 6-Param Search...")
        
        target_E = constraint_total_energy if constraint_total_energy is not None else (E0 * 1000.0)
        slope_target = -(rho0 * (D_cj**2) * 1e-9)
        
        # Define objective with high MSE weight
        def penalty_obj(p):
             # p: [A, B, R1, R2, w, C]
             return loss_penalty_jwl(p, V_rel_array, P_array, V_cj_rel, P_cj, target_E, Slope_target=slope_target)

        # Search space for R1, R2, w
        # We will use these to generate A, B, C seeds
        lb_r = jnp.array([3.5, 0.5, 0.25])
        ub_r = jnp.array([10.0, 3.5, 0.45])
        
        # Bounds for full 6 params
        lb = jnp.array([50.0, 0.1, 3.5, 0.5, 0.25, 0.1])
        ub = jnp.array([4000.0, 100.0, 10.0, 3.5, 0.45, 20.0])
        
        pso = PSOOptimizer(penalty_obj, lb, ub, num_particles=100)
        best_x, best_f = pso.optimize(num_iterations=400)
        
        # [V10.6] Nelder-Mead Refinement for sub-Gpa precision
        x0 = np.array(best_x)
        res = minimize(objective, x0, method='Nelder-Mead', tol=1e-6, options={'maxiter': 2000})
        A, B, R1, R2, w, C = res.x
        
        # Calculate final MSE
        P_final = A * np.exp(-R1 * V_rel_array) + \
                 B * np.exp(-R2 * V_rel_array) + \
                 C / (V_rel_array**(1.0 + w))
        final_mse = np.mean((np.log(np.maximum(P_final, 1e-6)) - y_log)**2)
        
        return JWLParams(A=float(A), B=float(B), R1=float(R1), R2=float(R2), omega=float(w), E0=float(E0), fit_mse=float(final_mse))



    if method == 'PSO':
        from pdu.calibration.pso import PSOOptimizer
        print(f"[V10 PSO] Running Global Optimization for JWL fitting...")
        
        # Define JAX-compatible objective for PSO
        def pso_obj(p):
            # p: [A, B, R1, R2, w, C]
            return objective(p)
            
        lb = jnp.array([10.0, 0.1, 1.5, 0.2, 0.05, 0.1])
        ub = jnp.array([5000.0, 300.0, 15.0, 5.0, 1.5, 50.0])
        
        pso = PSOOptimizer(pso_obj, lb, ub, num_particles=100)
        best_x_pso, best_f_pso = pso.optimize(num_iterations=150)
        x0 = np.array(best_x_pso)
        print(f"[V10 PSO] Global search result f={best_f_pso:.6f}. Refining with Nelder-Mead...")

    res = minimize(objective, x0, method='Nelder-Mead', tol=1e-6, options={'maxiter': 5000})
    
    A, B, R1, R2, w, C = res.x
    
    # 计算最终拟合误差 (Log-MSE)
    P_final = A * np.exp(-R1 * V_rel_array) + \
             B * np.exp(-R2 * V_rel_array) + \
             C / (V_rel_array**(1.0 + w))
    final_mse = np.mean((np.log(np.maximum(P_final, 1e-6)) - y_log)**2)
    
    return JWLParams(A=float(A), B=float(B), R1=float(R1), R2=float(R2), omega=float(w), E0=float(E0), fit_mse=float(final_mse))


# ==============================================================================
# V10.4 新增: CJ 锚定求解器 (P0-3 改进)
# ==============================================================================

def solve_jwl_linear_params(R1, R2, w, V_cj, P_cj, Slope_cj, E_cj):
    """
    解析求解 JWL 线性参数 A, B, C (V10.4)
    
    构建线性方程组 M * [A, B, C]^T = [P, Slope, E]^T
    
    Returns:
        A, B, C (all JAX scalars)
    """
    # 避免数值溢出 (Clamp instead of return None)
    # If overflow, matrix values dominate and result likely garbage/NaN, handled by caller
    
    term1 = jnp.exp(-R1 * V_cj)
    term2 = jnp.exp(-R2 * V_cj)
    term3_p = V_cj ** -(1.0 + w)
    term3_s = V_cj ** -(2.0 + w)
    term3_e = V_cj ** -w
    
    # 1. 压力方程 P(V) = A*t1 + B*t2 + C*t3_p
    row1 = [term1, term2, term3_p]
    
    # 2. 斜率方程 P'(V) = -A*R1*t1 - B*R2*t2 - C*(1+w)*t3_s
    row2 = [-R1 * term1, -R2 * term2, -(1.0 + w) * term3_s]
    
    # 3. 能量方程 E_int = A/R1*t1 + B/R2*t2 + C/w*t3_e
    # 注意: E_int 是从 V_cj 到无穷远的积分 (expansion work)
    # E_cj = internal energy at CJ state relative to expansion limit
    row3 = [term1 / R1, term2 / R2, term3_e / w]
    
    M = jnp.array([row1, row2, row3])
    y = jnp.array([P_cj, Slope_cj, E_cj])
    
    # 求解
    # remove try-except, rely on JAX returning NaNs for singular matrix
    sol = jnp.linalg.solve(M, y)
    return sol[0], sol[1], sol[2]




# ==============================================================================
# V10.5 新增: Relaxed Penalty Anchor (P0-5 改进)
# ==============================================================================

def loss_penalty_jwl(
    params, 
    V_data, P_data, 
    V_cj, P_cj, 
    E_target,
    Slope_target = None  # V10.6: Optional slope constraint
):
    """
    V10.6 Constrained Physics Loss (Response to Audit)
    Objective = LogMSE + Lambda_CJ*(P-P_cj)^2 + Lambda_Slope*(Slope-Slope_cj)^2 + Barriers
    """
    A, B, R1, R2, w, C = params
    
    # 1. Energy Constraint (Soft Penalty)
    term1 = (A / R1) * jnp.exp(-R1 * V_cj)
    term2 = (B / R2) * jnp.exp(-R2 * V_cj)
    term3 = (C / w) * (V_cj**-w)
    E_calc = term1 + term2 + term3
    loss_energy = ((E_calc - E_target) / E_target)**2 * 10000.0 # Increased weight
    
    # 2. Fit Error
    P_pred = A * jnp.exp(-R1 * V_data) + B * jnp.exp(-R2 * V_data) + C / (V_data**(1.0+w))
    penalty_neg_P = jnp.sum(jnp.where(P_pred < 1e-4, 1e9, 0.0))
    log_err = jnp.log(jnp.maximum(P_pred, 1e-6)) - jnp.log(jnp.maximum(P_data, 1e-4))
    mse = jnp.mean(log_err**2) * 5000.0
    
    # 3. CJ Anchor Penalty (V10.6 Harder Anchor)
    P_at_cj = A * jnp.exp(-R1 * V_cj) + B * jnp.exp(-R2 * V_cj) + C / (V_cj**(1.0+w))
    rel_err_cj = jnp.abs(P_at_cj - P_cj) / P_cj
    # V10.6: Abandon 5% relaxed zone, use 1% tolerance or direct penalty
    penalty_cj = jnp.where(rel_err_cj > 0.01, (rel_err_cj - 0.01)**2 * 200000.0, 0.0)
    
    # 3.1 Slope Anchor (Rayleigh Line consistency)
    loss_slope = 0.0
    if Slope_target is not None:
        dP_dV_fit = -A*R1*jnp.exp(-R1*V_cj) - B*R2*jnp.exp(-R2*V_cj) - C*(1.0+w)*(V_cj**-(2.0+w))
        # Relative slope error
        loss_slope = ((dP_dV_fit - Slope_target) / Slope_target)**2 * 500.0

    # 3.2 Sound Speed Stability watchdog (Ensure c2 > 0)
    def check_c2(v):
        dp_dv = -A*R1*jnp.exp(-R1*v) - B*R2*jnp.exp(-R2*v) - C*(1.0+w)*(v**-(2.0+w))
        return -v**2 * dp_dv
    
    v_test = jnp.array([1.0, 1.5, 2.5, 5.0, 10.0])
    c2_vals = jax.vmap(check_c2)(v_test)
    penalty_c2 = jnp.sum(jnp.where(c2_vals < 1e-4, 1e10, 0.0))

    # 4. [V10.6] Grüneisen Constraint (Physics Barrier)
    # Target range: [0.25, 0.45]
    w_min, w_max = 0.25, 0.45
    penalty_omega = jnp.where(w < w_min, (w_min - w)**2 * 100000.0, 0.0) + \
                    jnp.where(w > w_max, (w - w_max)**2 * 100000.0, 0.0)
    
    # [V10.6] Omega Centering: Favor values near 0.30 for physical topology
    penalty_w_center = (w - 0.30)**2 * 1000.0

    # 5. [V10.6] Topology Constraint (A vs B)
    # Prevent B from dominating high pressure. Usually A > 10*B
    ratio_BA = B / (A + 1e-3)
    penalty_topology = jnp.where(ratio_BA > 0.1, (ratio_BA - 0.1)**2 * 10000.0, 0.0)

    # 6. Physical Barriers (Monotonicity)
    barrier_A = jnp.where(A < 10.0, 1e8, 0.0)
    barrier_R = jnp.where(R1 < R2 + 0.8, 1e8, 0.0)
    
    return mse + penalty_cj + loss_energy + penalty_neg_P + barrier_A + barrier_R + penalty_omega + penalty_topology + loss_slope + penalty_c2 + penalty_w_center




