"""
热力学函数模块

基于 NASA 7系数或 9系数多项式计算热力学函数 (Cp, H, S, G)。
遵循混合精度策略：累加使用 FP64 保证精度。
"""

import jax
import jax.numpy as jnp
from typing import Union, Optional
from functools import partial

from pdu.utils.precision import R_GAS, to_fp64


@jax.jit
def compute_cp(coeffs: jnp.ndarray, T: float) -> float:
    """计算定压热容 Cp"""
    T = to_fp64(jnp.asarray(T))
    
    def _cp_7(c, t):
        return c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4

    def _cp_9(c, t):
        t_inv = 1.0 / t
        t_inv2 = t_inv * t_inv
        return c[0]*t_inv2 + c[1]*t_inv + c[2] + c[3]*t + c[4]*t**2 + c[5]*t**3 + c[6]*t**4

    if coeffs.shape[0] == 9:
        val = _cp_9(coeffs, T)
    else:
        val = _cp_7(coeffs, T)
        
    return R_GAS * val


@jax.jit
def compute_enthalpy(coeffs: jnp.ndarray, T: float) -> float:
    """计算摩尔焓 H"""
    T = to_fp64(jnp.asarray(T))
    
    def _h_7(c, t):
        return c[0] + c[1]*t/2.0 + c[2]*t**2/3.0 + c[3]*t**3/4.0 + c[4]*t**4/5.0 + c[5]/t

    def _h_9(c, t):
        t_inv = 1.0 / t
        t_inv2 = t_inv * t_inv
        term1 = -c[0] * t_inv2
        term2 = c[1] * t_inv * jnp.log(t)
        poly = c[2] + c[3]*t/2.0 + c[4]*t**2/3.0 + c[5]*t**3/4.0 + c[6]*t**4/5.0
        const = c[7] * t_inv
        return term1 + term2 + poly + const

    if coeffs.shape[0] == 9:
        h_over_rt = _h_9(coeffs, T)
    else:
        h_over_rt = _h_7(coeffs, T)
        
    return R_GAS * T * h_over_rt


@jax.jit
def compute_internal_energy(coeffs: jnp.ndarray, T: float) -> float:
    """计算摩尔内能 U"""
    H = compute_enthalpy(coeffs, T)
    T = to_fp64(jnp.asarray(T))
    return H - R_GAS * T

@jax.jit
def compute_entropy(coeffs: jnp.ndarray, T: float) -> float:
    """计算摩尔熵 S"""
    T = to_fp64(jnp.asarray(T))
    
    def _s_7(c, t):
        return c[0]*jnp.log(t) + c[1]*t + c[2]*t**2/2.0 + c[3]*t**3/3.0 + c[4]*t**4/4.0 + c[6]

    def _s_9(c, t):
        t_inv = 1.0 / t
        t_inv2 = t_inv * t_inv
        term1 = -c[0] * t_inv2 / 2.0
        term2 = -c[1] * t_inv
        log_term = c[2] * jnp.log(t)
        poly = c[3]*t + c[4]*t**2/2.0 + c[5]*t**3/3.0 + c[6]*t**4/4.0
        const = c[8]
        return term1 + term2 + log_term + poly + const

    if coeffs.shape[0] == 9:
        s_over_r = _s_9(coeffs, T)
    else:
        s_over_r = _s_7(coeffs, T)
        
    return R_GAS * s_over_r


@jax.jit  
def compute_gibbs(coeffs: jnp.ndarray, T: float, P: float = 1e5) -> float:
    """计算摩尔吉布斯自由能 G"""
    T = to_fp64(jnp.asarray(T))
    P = to_fp64(jnp.asarray(P))
    H = compute_enthalpy(coeffs, T)
    S = compute_entropy(coeffs, T)
    P0 = 1e5
    G = H - T * S + R_GAS * T * jnp.log(P / P0)
    return G


# Hoisted vmap for chemical potential
def _compute_chemical_potential_raw(coeffs, T, n_i, n_total, P):
    """Internal raw function for vmapping"""
    H = compute_enthalpy(coeffs, T)
    S = compute_entropy(coeffs, T)
    G0 = H - T * S
    x_i = n_i / (n_total + 1e-30)
    P0 = 1e5
    mu = G0 + R_GAS * T * jnp.log(jnp.maximum(x_i * P / P0, 1e-30))
    return mu

# Vmap over coeffs (0) and n_i (2), while T, n_total, P are constant (None)
_compute_chemical_potential_vec = jax.vmap(_compute_chemical_potential_raw, in_axes=(0, None, 0, None, None))

@jax.jit
def compute_chemical_potential(
    coeffs: jnp.ndarray, 
    T: float, 
    n_i: float, 
    n_total: float, 
    P: float = 1e5
) -> float:
    """计算化学势 μ_i"""
    T = to_fp64(jnp.asarray(T))
    n_i = to_fp64(jnp.asarray(n_i))
    n_total = to_fp64(jnp.asarray(n_total))
    P = to_fp64(jnp.asarray(P))
    return _compute_chemical_potential_raw(coeffs, T, n_i, n_total, P)


@partial(jax.jit, static_argnums=(2,))
def compute_gibbs_batch(
    coeffs_all: jnp.ndarray, 
    T: float,
    n_species: int,
    n: jnp.ndarray,
    P: float = 1e5
) -> jnp.ndarray:
    """批量计算所有物种的化学势"""
    T = to_fp64(jnp.asarray(T))
    n = to_fp64(n)
    P = to_fp64(jnp.asarray(P))
    n_total = jnp.sum(n) + 1e-30
    return _compute_chemical_potential_vec(coeffs_all, T, n, n_total, P)

