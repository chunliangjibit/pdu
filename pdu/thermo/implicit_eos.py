# pdu/thermo/implicit_eos.py
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import custom_jvp, jit, lax
from typing import Tuple, NamedTuple, Optional
from functools import partial

from pdu.core.equilibrium import solve_equilibrium
from pdu.physics.eos import compute_pressure_jcz3, compute_internal_energy_jcz3, compute_entropy_consistent
from pdu.utils.precision import to_fp64, to_fp32

"""
PDU V11.0 Tier 0: 隐式微分热力学内核 (极速一致性版)
"""

class ThermoState(NamedTuple):
    P: jnp.ndarray 
    T: jnp.ndarray 
    cs: jnp.ndarray 
    n: jnp.ndarray 

@custom_jvp
def get_thermo_properties(rho, T, atom_vec, coeffs_all, A_matrix, atomic_masses, eos_params, n_init=None):
    """
    给定 rho, T -> 计算平衡组分 n，返回 P, n
    支持 n_init (Warm-Start)
    """
    rho_64 = to_fp64(rho)
    T_64 = to_fp64(T)
    
    mw_g_per_mol = jnp.dot(atom_vec, atomic_masses)
    V_molar = mw_g_per_mol / (rho_64 + 1e-10)
    
    n_final = solve_equilibrium(atom_vec, V_molar, T_64, A_matrix, coeffs_all, eos_params, n_init)
    
    P_pa = compute_pressure_jcz3(
        n_final, V_molar, T_64, coeffs_all, 
        eos_params[0], eos_params[1], eos_params[2], eos_params[3],
        eos_params[4], eos_params[5], 
        eos_params[6], eos_params[7], eos_params[8],
        eos_params[9],
        mw_g_per_mol
    )
    
    return to_fp32(P_pa), to_fp32(n_final)

@get_thermo_properties.defjvp
def get_thermo_properties_jvp(primals, tangents):
    # n_init is currently treated as non-diff or zero-tan
    rho, T, atom_vec, coeffs_all, A_matrix, atomic_masses, eos_params, n_init = primals
    d_rho, d_T, d_atom_vec, d_coeffs_all, d_A_matrix, d_atomic_masses, d_eos_params, d_n_init = tangents
    
    P, n = get_thermo_properties(rho, T, atom_vec, coeffs_all, A_matrix, atomic_masses, eos_params, n_init)
    
    rho_64 = to_fp64(rho)
    T_64 = to_fp64(T)
    mw_g_per_mol = jnp.dot(atom_vec, atomic_masses)
    n_64 = to_fp64(n)

    def p_func(r, t):
        v = mw_g_per_mol / (r + 1e-10)
        return compute_pressure_jcz3(n_64, v, t, coeffs_all, *eos_params, mw_g_per_mol)

    dP_drho, dP_dT = jax.grad(p_func, argnums=(0, 1))(rho_64, T_64)
    dP_total = dP_drho * d_rho + dP_dT * d_T
    
    return (P, n), (to_fp32(dP_total), jnp.zeros_like(n))

@jit
def get_sound_speed(rho, T, n, atom_vec, coeffs_all, eos_params, atomic_masses):
    """
    计算冻结声速 (Frozen Sound Speed)
    a^2 = (dP/drho)_s = (dP/drho)_T + (dP/dT)_rho * (T * (dP/dT)_rho) / (rho^2 * Cv)
    """
    rho_64 = to_fp64(rho)
    T_64 = to_fp64(T)
    n_64 = to_fp64(n)
    mw_g_per_mol = jnp.dot(atom_vec, atomic_masses)
    
    def p_func(r, t):
        v = mw_g_per_mol / (r + 1e-10)
        return compute_pressure_jcz3(n_64, v, t, coeffs_all, *eos_params, mw_g_per_mol)
    
    def u_func(r, t):
        v = mw_g_per_mol / (r + 1e-10)
        return compute_internal_energy_jcz3(n_64, v, t, coeffs_all, *eos_params, mw_g_per_mol)

    # 1. 计算偏导数
    dP_drho, dP_dT = jax.grad(p_func, argnums=(0, 1))(rho_64, T_64)
    dU_dT = jax.grad(u_func, argnums=1)(rho_64, T_64)
    
    # 2. 定容比热 Cv (J/kg*K)
    mw_kg = mw_g_per_mol / 1000.0
    cv = dU_dT / mw_kg
    
    # 3. 绝热声速
    # a^2 = dP/drho + (T/rho^2/Cv) * (dP/dT)^2 ? 
    # 单位转换注意：rho 是 g/cm3 -> 1000 kg/m3
    rho_si = rho_64 * 1000.0
    term_entropy = (T_64 / (rho_si**2 * cv + 1e-10)) * (dP_dT**2)
    
    # dP_drho 这里的 rho 是 g/cm3, 需要转为 kg/m3
    # P(Pa), rho(kg/m3) => a^2 (m2/s2)
    # dP/drho_si = dP/drho_gcm3 / 1000
    a2 = (dP_drho / 1000.0) + term_entropy
    
    return to_fp32(jnp.sqrt(jnp.maximum(a2, 1e-6)))

@jit
def get_internal_energy_pt(rho, T, atom_vec, coeffs_all, A_matrix, atomic_masses, eos_params):
    rho_64 = to_fp64(rho)
    T_64 = to_fp64(T)
    mw_g_per_mol = jnp.dot(atom_vec, atomic_masses)
    V_molar = mw_g_per_mol / (rho_64 + 1e-10)
    
    n_eq = solve_equilibrium(atom_vec, V_molar, T_64, A_matrix, coeffs_all, eos_params)
    u_mol = compute_internal_energy_jcz3(n_eq, V_molar, T_64, coeffs_all, *eos_params, mw_g_per_mol)
    
    return to_fp32(u_mol / (mw_g_per_mol / 1000.0))