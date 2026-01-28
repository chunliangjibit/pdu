
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

def fit_jwl_from_isentrope(V_rel_array, P_array, rho0, E0):
    """
    V8: Robust Log-Space Fitting with CJ Anchoring
    
    采用对数域损失函数均衡高低压误差，并强制锚定 CJ 点起点的物理一致性。
    """
    from scipy.optimize import minimize
    import numpy as np

    # 1. 锚点数据 (CJ 点)
    V_cj_rel = V_rel_array[0]
    P_cj = P_array[0]
    
    # 2. 对数域数据 (用于平衡 30GPa 和 0.1GPa 的权重)
    y_log = np.log(np.maximum(P_array, 1e-4))
    
    # 3. 目标函数
    def objective(params):
        A, B, R1, R2, w, C = params
        
        # 物理约束惩罚 (Barrier Method)
        # 强制 R1 > R2 + 0.5 防止模态合并；强制 w 在物理区间
        if A < 0 or B < 0 or C < 0 or w < 0 or w > 1.2 or R1 < (R2 + 0.5) or R2 < 0.1:
            return 1e9
        
        # 模型预测 (Isentrope 形式)
        P_pred = A * np.exp(-R1 * V_rel_array) + \
                 B * np.exp(-R2 * V_rel_array) + \
                 C / (V_rel_array**(1.0 + w))
        
        y_pred_log = np.log(np.maximum(P_pred, 1e-6))
        
        # Log-MSE: 使得低压段的相对误差与高压段同等重要
        loss_fit = np.mean((y_pred_log - y_log)**2)
        
        # CJ Anchor Penalty: 强行拉回起点，保证爆轰瞬态物理正确
        P_cj_pred = A * np.exp(-R1 * V_cj_rel) + \
                    B * np.exp(-R2 * V_cj_rel) + \
                    C / (V_cj_rel**(1.0 + w))
        loss_anchor = ((P_cj_pred - P_cj) / P_cj) ** 2 * 100.0 # 100倍权重锚点
        
        return loss_fit + loss_anchor

    # 4. 优化准备 (初始猜测)
    # A, B, R1, R2, w, C
    x0 = [600.0, 15.0, 4.8, 1.2, 0.35, 1.0]
    
    res = minimize(objective, x0, method='Nelder-Mead', tol=1e-5, options={'maxiter': 2000})
    
    A, B, R1, R2, w, C = res.x
    
    # 返回标准 JWL 参数结构
    return JWLParams(A=float(A), B=float(B), R1=float(R1), R2=float(R2), omega=float(w), E0=float(E0))

