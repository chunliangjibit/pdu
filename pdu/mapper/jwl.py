# pdu/mapper/jwl.py
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

from pdu.core.types import GasState, ParticleState, State
from pdu.physics.jwl import fit_jwl_from_isentrope
from pdu.thermo.implicit_eos import get_balanced_thermo

"""
PDU V11.0 P4: JWL 投影器 (Mapper)
将微观两相流场（及随后的等熵膨胀）投影为宏观工程 JWL 参数。
"""

def compute_total_pressure(gas, part):
    """
    计算总压力 (GPa)： P_tot = P_gas + Momentum_Flux
    """
    # 此处假设 gas.P (Pa) 已经计算好。如果没有，需调用 EOS
    # 为了演示，我们假设 P_gas 是从外部传入或已在 aux 中。
    # 暂定简化公式：
    # P_stagnation = P_gas + sum(phi_k * rho_k * (u_g - u_p,k)^2)
    
    # 注意单位：rho_k (g/cm3) -> kg/m3 (*1000), u (m/s), P_gas (Pa)
    slip_u = gas.u - part.u
    momentum_flux = jnp.sum(part.phi * (part.rho * 1000.0) * (slip_u**2))
    
    # 转换为 GPa
    p_tot_gpa = (gas.P + momentum_flux) / 1e9
    return p_tot_gpa

def calculate_mixture_isentrope(cj_state, rho0, n_points=30):
    """
    从 CJ 点出发，运行等熵膨胀过程，生成 P-V 数据点。
    注意：在膨胀段通常假设两相达到热力学平衡。
    """
    # 1. 初始比容 V/V0 = 1.0 (相对于装药密度)
    V_start = 1.0 / (rho0 / cj_state.gas.rho) # 简化
    # 实际上应该是 V_cj / V0
    
    # 为了简化 P4 演示，我们生成一条模拟等熵线
    v_ratios = jnp.linspace(1.0, 8.0, n_points)
    # 假设伽马律： P = P_cj * (V/V_cj)^-gamma
    gamma = 3.0
    p_points = (cj_state.P_tot) * (v_ratios**(-gamma))
    
    return v_ratios, p_points

def project_to_jwl(cj_info, rho0, E0_theory):
    """
    核心投影接口：
    cj_info: 包含 D, P_cj (total), rho_cj 等
    E0_theory: 理论总爆热 (GPa)
    """
    # 1. 生成等熵线数据
    V_rel, P_data = calculate_mixture_isentrope(cj_info, rho0)
    
    # 2. 调用 V10 拟合引擎 (包含物理围栏)
    # fit_jwl_from_isentrope(V_rel, P_array, rho0, E0, D_cj, P_cj_theory)
    jwl_params = fit_jwl_from_isentrope(
        V_rel, P_data, rho0, E0_theory, 
        cj_info.D, cj_info.P_tot,
        method='RELAXED_PENALTY'
    )
    
    return jwl_params
