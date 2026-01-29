
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
        
        # ========== V10.2 增强物理约束 ==========
        # 基础参数范围约束
        bad = (A < 0) | (B < 0) | (C < 0) | (w < 0) | (w > 1.2) | (R1 < (R2 + 0.5)) | (R2 < 0.1)
        penalty_barrier = jnp.where(bad, 1e9, 0.0)
        
        # (F) 声速正定性检查: c² = -V² * dP/dV > 0
        def check_sound_speed(V):
            dP_dV = -A * R1 * jnp.exp(-R1 * V) - \
                    B * R2 * jnp.exp(-R2 * V) - \
                    C * (1.0 + w) * (V**-(2.0 + w))
            c_squared = -V**2 * dP_dV
            return c_squared
        
        # 在多个体积点检查声速
        V_check_points = jnp.array([0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
        c2_values = jax.vmap(check_sound_speed)(V_check_points)
        sound_speed_penalty = jnp.sum(jnp.where(c2_values < 0, 1e8, 0.0))
        
        # (G) 曲率约束: d²P/dV² > 0 (凸性)
        def check_curvature(V):
            d2P_dV2 = A * R1**2 * jnp.exp(-R1 * V) + \
                      B * R2**2 * jnp.exp(-R2 * V) + \
                      C * (1.0 + w) * (2.0 + w) * (V**-(3.0 + w))
            return d2P_dV2
        
        curvature_values = jax.vmap(check_curvature)(V_check_points)
        curvature_penalty = jnp.sum(jnp.where(curvature_values < 0, 1e7, 0.0))
        
        # ========== 原有约束 ==========
        # 模型预测 (Isentrope 形式)
        P_pred = A * jnp.exp(-R1 * V_rel_array) + \
                 B * jnp.exp(-R2 * V_rel_array) + \
                 C / (jnp.maximum(V_rel_array, 1e-4)**(1.0 + w))
        
        y_pred_log = jnp.log(jnp.maximum(P_pred, 1e-6))
        
        # (A) Log-MSE Loss - V10.2: 分段权重
        # 高压区 (V < 1.2): 权重 2.0
        # 中压区 (1.2 <= V < 3.0): 权重 1.5
        # 低压区 (V >= 3.0): 权重 1.0
        weights = jnp.where(V_rel_array < 1.2, 2.0,
                           jnp.where(V_rel_array < 3.0, 1.5, 1.0))
        loss_fit = jnp.mean(weights * (y_pred_log - y_log)**2) * 50.0
        
        # (B) P_CJ Anchor Penalty
        if target_P_cj is not None:
             P_fit_cj = A * jnp.exp(-R1 * target_V_cj) + \
                       B * jnp.exp(-R2 * target_V_cj) + \
                       C / (jnp.maximum(target_V_cj, 1e-4)**(1.0 + w))
             loss_anchor_P = ((P_fit_cj - target_P_cj) / target_P_cj) ** 2 * 10000.0 
        else:
             P_fit_cj = A * jnp.exp(-R1 * V_cj_rel) + \
                       B * jnp.exp(-R2 * V_cj_rel) + \
                       C / (jnp.maximum(V_cj_rel, 1e-4)**(1.0 + w))
             loss_anchor_P = ((P_fit_cj - P_cj) / P_cj) ** 2 * 100.0
        
        # (C) Gamma_CJ Anchor Penalty
        if target_P_cj is None:
            dP_dV = -A * R1 * jnp.exp(-R1 * V_cj_rel) - \
                    B * R2 * jnp.exp(-R2 * V_cj_rel) - \
                    C * (1.0 + w) * (V_cj_rel**-(2.0 + w))
            gamma_pred = -(V_cj_rel / (P_fit_cj + 1e-6)) * dP_dV
            loss_anchor_Gamma = ((gamma_pred - gamma_cj_target) / gamma_cj_target) ** 2 * 50.0 
        else:
            loss_anchor_Gamma = 0.0
            
        # (D) Total Energy Constraint [V8.7 Upgrade]
        loss_energy = 0.0
        if constraint_total_energy is not None:
            V_start = target_V_cj if target_V_cj is not None else V_cj_rel
            
            term1 = (A / R1) * jnp.exp(-R1 * V_start)
            term2 = (B / R2) * jnp.exp(-R2 * V_start)
            term3 = (C / jnp.maximum(w, 1e-3)) * (V_start**(-w))
            
            E_integral = term1 + term2 + term3
            loss_energy = ((E_integral - constraint_total_energy) / constraint_total_energy) ** 2 * 5000.0
        
        # (H) V10.2 膨胀功分段验证
        # 计算高压区 (V: 0.6-1.5) 和中压区 (V: 1.5-5.0) 的膨胀功
        V_high = jnp.array([0.6, 0.8, 1.0, 1.2, 1.5])
        V_mid = jnp.array([1.5, 2.0, 3.0, 4.0, 5.0])
        
        P_high = A * jnp.exp(-R1 * V_high) + B * jnp.exp(-R2 * V_high) + C / (V_high**(1.0 + w))
        P_mid = A * jnp.exp(-R1 * V_mid) + B * jnp.exp(-R2 * V_mid) + C / (V_mid**(1.0 + w))
        
        # 膨胀功应随体积增加而增加 (积分值递增)
        work_high = jnp.trapezoid(P_high, V_high)
        work_mid = jnp.trapezoid(P_mid, V_mid)
        
        # 高压区贡献应大于中压区
        work_penalty = jnp.where(work_high < work_mid * 0.5, 1e6, 0.0)
            
        # (E) Prior Penalty
        loss_prior = 0.0
        if exp_priors:
            ep = exp_priors
            loss_prior += ((A - ep['A'])/ep['A'])**2 * 5.0
            loss_prior += ((B - ep['B'])/ep['B'])**2 * 2.0
            loss_prior += ((R1 - ep['R1'])/ep['R1'])**2 * 10.0
            loss_prior += ((R2 - ep['R2'])/ep['R2'])**2 * 10.0
            loss_prior += ((w - ep['omega'])/ep['omega'])**2 * 10.0
            
        return (loss_fit + loss_anchor_P + loss_anchor_Gamma + loss_energy + loss_prior + 
                penalty_barrier + sound_speed_penalty + curvature_penalty + work_penalty)

    # 5. 优化准备 (使用先验作为初始值)
    if exp_priors:
        ep = exp_priors
        x0 = [ep['A'], ep['B'], ep['R1'], ep['R2'], ep['omega'], 1.0]
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


