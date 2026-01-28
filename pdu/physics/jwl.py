
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

def fit_jwl_from_isentrope(V_rel_array, P_array, rho0, E0, D_cj, P_cj_theory, exp_priors=None):
    """
    V8.5: Prior-Aware Robust Fitting
    
    增加文献先验约束，平衡数学解的冗余性，优先寻找靠近标准 Basin 的物理参数。
    """
    from scipy.optimize import minimize
    import numpy as np

    # 1. 物理锚点数据 (CJ 点)
    V_cj_rel = V_rel_array[0]
    P_cj = P_array[0]
    
    # 2. 计算物理目标 Gamma (绝热指数)
    gamma_cj_target = (rho0 * (D_cj**2) * 1e-6) / P_cj_theory - 1.0 
    
    # 3. 对数域数据
    y_log = np.log(np.maximum(P_array, 1e-4))
    
    # 4. 目标函数
    def objective(params):
        A, B, R1, R2, w, C = params
        
        # 物理约束惩罚 (Barrier Method)
        if A < 0 or B < 0 or C < 0 or w < 0 or w > 1.2 or R1 < (R2 + 1.0) or R2 < 0.1:
            return 1e9
        
        # 模型预测 (Isentrope 形式)
        P_pred = A * np.exp(-R1 * V_rel_array) + \
                 B * np.exp(-R2 * V_rel_array) + \
                 C / (V_rel_array**(1.0 + w))
        
        y_pred_log = np.log(np.maximum(P_pred, 1e-6))
        
        # (A) Log-MSE Loss (数据拟合) - 权重降低，给先验留空间
        loss_fit = np.mean((y_pred_log - y_log)**2) * 50.0
        
        # (B) P_CJ Anchor Penalty
        P_fit_cj = A * np.exp(-R1 * V_cj_rel) + \
                  B * np.exp(-R2 * V_cj_rel) + \
                  C / (V_cj_rel**(1.0 + w))
        loss_anchor_P = ((P_fit_cj - P_cj) / P_cj) ** 2 * 100.0
        
        # (C) Gamma_CJ Anchor Penalty
        dP_dV = -A * R1 * np.exp(-R1 * V_cj_rel) - \
                B * R2 * np.exp(-R2 * V_cj_rel) - \
                C * (1.0 + w) * (V_cj_rel**-(2.0 + w))
        gamma_pred = -(V_cj_rel / (P_fit_cj + 1e-6)) * dP_dV
        loss_anchor_Gamma = ((gamma_pred - gamma_cj_target) / gamma_cj_target) ** 2 * 50.0 
        
        # (D) Prior Penalty (文献对标先验)
        loss_prior = 0.0
        if exp_priors:
            ep = exp_priors
            loss_prior += ((A - ep['A'])/ep['A'])**2 * 5.0
            loss_prior += ((B - ep['B'])/ep['B'])**2 * 2.0
            loss_prior += ((R1 - ep['R1'])/ep['R1'])**2 * 10.0
            loss_prior += ((R2 - ep['R2'])/ep['R2'])**2 * 10.0
            loss_prior += ((w - ep['omega'])/ep['omega'])**2 * 10.0
            
        return loss_fit + loss_anchor_P + loss_anchor_Gamma + loss_prior

    # 5. 优化准备 (使用先验作为初始值)
    if exp_priors:
        ep = exp_priors
        x0 = [ep['A'], ep['B'], ep['R1'], ep['R2'], ep['omega'], 1.0]
    else:
        x0 = [600.0, 15.0, 4.8, 1.2, 0.35, 1.0]
    
    res = minimize(objective, x0, method='Nelder-Mead', tol=1e-6, options={'maxiter': 5000})
    
    A, B, R1, R2, w, C = res.x
    
    # 计算最终拟合误差 (Log-MSE)
    P_final = A * np.exp(-R1 * V_rel_array) + \
             B * np.exp(-R2 * V_rel_array) + \
             C / (V_rel_array**(1.0 + w))
    final_mse = np.mean((np.log(np.maximum(P_final, 1e-6)) - y_log)**2)
    
    return JWLParams(A=float(A), B=float(B), R1=float(R1), R2=float(R2), omega=float(w), E0=float(E0), fit_mse=float(final_mse))

