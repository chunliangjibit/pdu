"""
热力学函数模块

基于 NASA 7系数多项式计算热力学函数 (Cp, H, S, G)。
遵循混合精度策略：累加使用 FP64 保证精度。
"""

import jax
import jax.numpy as jnp
from typing import Union, Optional
from functools import partial

from pdu.utils.precision import R_GAS, to_fp64


@jax.jit
def compute_cp(coeffs: jnp.ndarray, T: float) -> float:
    """计算定压热容 Cp
    
    NASA 多项式:
    Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
    
    Args:
        coeffs: NASA 7系数 [a1, a2, a3, a4, a5, a6, a7]
        T: 温度 (K)
        
    Returns:
        Cp (J/(mol·K))
    """
    T = to_fp64(jnp.asarray(T))
    a1, a2, a3, a4, a5, a6, a7 = coeffs
    
    cp_over_r = a1 + a2*T + a3*T**2 + a4*T**3 + a5*T**4
    
    return R_GAS * cp_over_r


@jax.jit
def compute_enthalpy(coeffs: jnp.ndarray, T: float) -> float:
    """计算摩尔焓 H
    
    NASA 多项式:
    H/(R*T) = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T
    
    Args:
        coeffs: NASA 7系数 [a1, a2, a3, a4, a5, a6, a7]
        T: 温度 (K)
        
    Returns:
        H (J/mol)
    """
    T = to_fp64(jnp.asarray(T))
    a1, a2, a3, a4, a5, a6, a7 = coeffs
    
    h_over_rt = (a1 + a2*T/2.0 + a3*T**2/3.0 + a4*T**3/4.0 + 
                 a5*T**4/5.0 + a6/T)
    
    return R_GAS * T * h_over_rt


@jax.jit
def compute_internal_energy(coeffs: jnp.ndarray, T: float) -> float:
    """计算摩尔内能 U
    
    U = H - PV = H - RT (对于理想气体)
    
    Args:
        coeffs: NASA 7系数
        T: 温度 (K)
        
    Returns:
        U (J/mol)
    """
    H = compute_enthalpy(coeffs, T)
    T = to_fp64(jnp.asarray(T))
    # U = H - RT
    return H - R_GAS * T

@jax.jit
def compute_entropy(coeffs: jnp.ndarray, T: float) -> float:
    """计算摩尔熵 S
    
    NASA 多项式:
    S/R = a1*ln(T) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7
    
    Args:
        coeffs: NASA 7系数 [a1, a2, a3, a4, a5, a6, a7]
        T: 温度 (K)
        
    Returns:
        S (J/(mol·K))
    """
    T = to_fp64(jnp.asarray(T))
    a1, a2, a3, a4, a5, a6, a7 = coeffs
    
    s_over_r = (a1*jnp.log(T) + a2*T + a3*T**2/2.0 + 
                a4*T**3/3.0 + a5*T**4/4.0 + a7)
    
    return R_GAS * s_over_r


@jax.jit  
def compute_gibbs(coeffs: jnp.ndarray, T: float, P: float = 1e5) -> float:
    """计算摩尔吉布斯自由能 G
    
    G = H - T*S + R*T*ln(P/P0)
    
    对于理想气体，包含压力修正项。
    
    Args:
        coeffs: NASA 7系数 [a1, a2, a3, a4, a5, a6, a7]
        T: 温度 (K)
        P: 压力 (Pa)，默认 1e5 (1 bar)
        
    Returns:
        G (J/mol)
    """
    T = to_fp64(jnp.asarray(T))
    P = to_fp64(jnp.asarray(P))
    
    H = compute_enthalpy(coeffs, T)
    S = compute_entropy(coeffs, T)
    
    # 标准压力 P0 = 1 bar = 1e5 Pa
    P0 = 1e5
    
    G = H - T * S + R_GAS * T * jnp.log(P / P0)
    
    return G


@jax.jit
def compute_chemical_potential(
    coeffs: jnp.ndarray, 
    T: float, 
    n_i: float, 
    n_total: float, 
    P: float = 1e5
) -> float:
    """计算化学势 μ_i
    
    对于理想气体混合物:
    μ_i = G_i^0(T) + R*T*ln(x_i * P / P0)
    
    其中 x_i = n_i / n_total 是摩尔分数
    
    Args:
        coeffs: NASA 7系数
        T: 温度 (K)
        n_i: 组分 i 的摩尔数
        n_total: 总摩尔数
        P: 压力 (Pa)
        
    Returns:
        μ_i (J/mol)
    """
    T = to_fp64(jnp.asarray(T))
    n_i = to_fp64(jnp.asarray(n_i))
    n_total = to_fp64(jnp.asarray(n_total))
    P = to_fp64(jnp.asarray(P))
    
    # 标准态吉布斯能
    H = compute_enthalpy(coeffs, T)
    S = compute_entropy(coeffs, T)
    G0 = H - T * S
    
    # 摩尔分数（加小量避免 log(0)）
    x_i = n_i / (n_total + 1e-30)
    
    # 标准压力
    P0 = 1e5
    
    # 化学势
    mu = G0 + R_GAS * T * jnp.log(jnp.maximum(x_i * P / P0, 1e-30))
    
    return mu


@partial(jax.jit, static_argnums=(2,))
def compute_gibbs_batch(
    coeffs_all: jnp.ndarray, 
    T: float,
    n_species: int,
    n: jnp.ndarray,
    P: float = 1e5
) -> jnp.ndarray:
    """批量计算所有物种的化学势
    
    Args:
        coeffs_all: 所有物种的 NASA 系数 (n_species, 7)
        T: 温度 (K)
        n_species: 物种数量
        n: 各物种摩尔数 (n_species,)
        P: 压力 (Pa)
        
    Returns:
        化学势数组 (n_species,) (J/mol)
    """
    T = to_fp64(jnp.asarray(T))
    n = to_fp64(n)
    P = to_fp64(jnp.asarray(P))
    
    n_total = jnp.sum(n) + 1e-30
    
    def compute_single_mu(coeffs, n_i):
        return compute_chemical_potential(coeffs, T, n_i, n_total, P)
    
    return jax.vmap(compute_single_mu)(coeffs_all, n)


@jax.jit
def compute_total_gibbs(
    coeffs_all: jnp.ndarray,
    T: float,
    n: jnp.ndarray,
    P: float = 1e5
) -> float:
    """计算系统总吉布斯自由能
    
    G_total = Σ n_i * μ_i
    
    Args:
        coeffs_all: 所有物种的 NASA 系数 (n_species, 7)
        T: 温度 (K)
        n: 各物种摩尔数 (n_species,)
        P: 压力 (Pa)
        
    Returns:
        G_total (J)
    """
    n = to_fp64(n)
    mu = compute_gibbs_batch(coeffs_all, T, n.shape[0], n, P)
    
    return jnp.sum(n * mu)


@jax.jit
def compute_total_enthalpy(
    coeffs_all: jnp.ndarray,
    T: float,
    n: jnp.ndarray
) -> float:
    """计算系统总焓
    
    H_total = Σ n_i * H_i
    
    Args:
        coeffs_all: 所有物种的 NASA 系数 (n_species, 7)
        T: 温度 (K)
        n: 各物种摩尔数 (n_species,)
        
    Returns:
        H_total (J)
    """
    n = to_fp64(n)
    
    def compute_h(coeffs):
        return compute_enthalpy(coeffs, T)
    
    H_all = jax.vmap(compute_h)(coeffs_all)
    
    return jnp.sum(n * H_all)


def get_species_coeffs(species_list: list, T: float) -> jnp.ndarray:
    """获取物种列表的 NASA 系数矩阵
    
    根据温度选择高温或低温系数。
    
    Args:
        species_list: 物种名称列表
        T: 温度 (K)
        
    Returns:
        系数矩阵 (n_species, 7)
    """
    from pdu.data.products import get_product_thermo
    
    coeffs_list = []
    for species in species_list:
        prod = get_product_thermo(species)
        coeffs = prod.get_coeffs(T)
        coeffs_list.append(coeffs)
    
    return jnp.stack(coeffs_list, axis=0)
