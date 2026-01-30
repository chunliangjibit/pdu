# pdu/thermo/implicit_eos.py
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import custom_jvp, jit, lax
from typing import Tuple, NamedTuple, Optional
from functools import partial

from pdu.core.equilibrium import solve_equilibrium
from pdu.physics.eos import compute_pressure_jcz3, compute_internal_energy_jcz3, compute_entropy_consistent, compute_total_helmholtz_energy, smooth_floor
from pdu.utils.precision import to_fp64, to_fp32

# Debug switch
_DEBUG_EOS_UNITS = False

@jit
def _atomic_masses_to_kg_per_mol(atomic_masses):
    """
    Accept atomic masses in either:
      - g/mol  (typical magnitudes: 1..238)
      - kg/mol (typical magnitudes: 1e-3..0.238)
    Convert to kg/mol.
    """
    am = jnp.asarray(atomic_masses)
    # If max is > 0.5, it's almost certainly g/mol.
    return jnp.where(jnp.max(am) > 0.5, am * 1e-3, am)

@jit
def _rho_to_kg_per_m3(rho):
    """
    Accept rho in either:
      - g/cm3 (typical condensed: 0.5..20)
      - kg/m3 (typical condensed: 500..20000)
    Convert to kg/m3.
    """
    r = jnp.asarray(rho)
    return jnp.where(r > 50.0, r, r * 1000.0)

@jit
def _mixture_mass_and_mw_avg_kg_per_mol(n, atom_vec, atomic_masses):
    """
    Returns:
      m_kg: total mass of mixture (kg) - derived from atom_vec (system invariants)
      mw_avg_kg_per_mol: average molecular weight (kg/mol)
      mw_species_kg_per_mol: species molecular weights (kg/mol)
      n_tot: total moles (mol)
    """
    am_kg_per_mol = _atomic_masses_to_kg_per_mol(atomic_masses)
    
@jit
def _mixture_mass_and_mw_avg_kg_per_mol(n, atom_vec, atomic_masses):
    """
    Returns:
      m_kg: total mass of mixture (kg) - derived from atom_vec (system invariants)
      mw_avg_kg_per_mol: average molecular weight (kg/mol)
      n_tot: total moles (mol)
    """
    am_kg_per_mol = _atomic_masses_to_kg_per_mol(atomic_masses)
    
    # atom_vec is (natoms,) representing the system formula (e.g. C4H8N8O8)
    # Total mass is fixed by this vector regardless of speciation n
    m_kg = jnp.dot(atom_vec, am_kg_per_mol) # Scalar
    
    n = jnp.asarray(n)
    n_tot = jnp.sum(n)
    mw_avg_kg_per_mol = m_kg / (n_tot + 1e-300)

    # Note: We don't verify speciation mass here, we trust conservation.
    return m_kg, mw_avg_kg_per_mol, n_tot


"""
PDU V11.0 Tier 0: 隐式微分热力学内核 (极速一致性版)
"""

class ThermoState(NamedTuple):
    P: jnp.ndarray 
    T: jnp.ndarray 
    cs: jnp.ndarray 
    n: jnp.ndarray 

@custom_jvp
def get_thermo_properties(rho, T, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params, n_init=None):
    """
    给定 rho, T -> 计算平衡组分 n，返回 P, n
    """
    rho_64 = to_fp64(rho)
    T_64 = to_fp64(T)
    
    # --- Unit Normalization ---
    rho_kg_m3 = _rho_to_kg_per_m3(rho_64)
    am_kg_per_mol = _atomic_masses_to_kg_per_mol(atomic_masses)
    
    # Use CGS for legacy solver input
    am_g_per_mol = am_kg_per_mol * 1000.0
    mw_g_per_mol = jnp.dot(atom_vec, am_g_per_mol) # Vector of species MWs
    rho_cgs = rho_kg_m3 * 1e-3
    
    V_molar = mw_g_per_mol / (rho_cgs + 1e-10) # cm3 / mol_species
    
    if _DEBUG_EOS_UNITS:
        jax.debug.print("SOLVER_INPUT: mw_g={} g/mol, rho_cgs={} g/cm3, V_molar={} cm3/mol, T={}", mw_g_per_mol, rho_cgs, V_molar, T_64)
    
    n_final = solve_equilibrium(atom_vec, V_molar, T_64, A_matrix, coeffs_low, coeffs_high, eos_params, n_init)
    
    # --- Now we have n, we can compute P using the ROBUST path ---
    m_kg, mw_avg_kg, n_tot = _mixture_mass_and_mw_avg_kg_per_mol(n_final, atom_vec, atomic_masses)
    
    V_total_m3 = m_kg / rho_kg_m3 # m3
    V_molar_cm3 = (V_total_m3 / (n_tot + 1e-300)) * 1e6
    
    if _DEBUG_EOS_UNITS:
        jax.debug.print(
            "THERMO_PROP: rho_in={} g/cm3, rho_SI={} kg/m3, n_tot={} mol, m_kg={} kg, V_tot_m3={} m3, V_mol_cm3={} cm3/mol",
            rho, rho_kg_m3, n_tot, m_kg, V_total_m3, V_molar_cm3
        )

    # Use compute_total_helmholtz_energy to get consistent pressure P = -dA/dV
    P_pa = compute_pressure_jcz3(
        n_final, V_molar, T_64, coeffs_low, coeffs_high,
        eos_params[0], eos_params[1], eos_params[2], eos_params[3],
        eos_params[4], eos_params[5], 
        eos_params[6], eos_params[7], eos_params[8],
        eos_params[9],
        mw_g_per_mol # Ensure this is g/mol
    )
    
    return to_fp32(P_pa), to_fp32(n_final)

@get_thermo_properties.defjvp
def get_thermo_properties_jvp(primals, tangents):
    # n_init is currently treated as non-diff or zero-tan
    rho, T, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params, n_init = primals
    d_rho, d_T, d_atom_vec, d_cl, d_ch, d_A, d_masses, d_eos, d_ni = tangents
    
    P, n = get_thermo_properties(rho, T, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params, n_init)
    
    rho_64 = to_fp64(rho)
    T_64 = to_fp64(T)
    mw_g_per_mol = jnp.dot(atom_vec, atomic_masses)
    n_64 = to_fp64(n)

    def p_func(r, t):
        v = mw_g_per_mol / (r + 1e-10)
        return compute_pressure_jcz3(n_64, v, t, coeffs_low, coeffs_high, *eos_params, mw_g_per_mol)

    dP_drho, dP_dT = jax.grad(p_func, argnums=(0, 1))(rho_64, T_64)
    dP_total = dP_drho * d_rho + dP_dT * d_T
    
    return (P, n), (to_fp32(dP_total), jnp.zeros_like(n))

@jit
def get_sound_speed(rho, T, n, atom_vec, coeffs_low, coeffs_high, eos_params, atomic_masses):
    """
    计算冻结声速 (Consistent A-only Derivation)
    Patch C: 只从 Helmholtz A(V,T) 推导声速，确保热力学一致性
    """
    # 冻结组成，避免 AD 污染
    n = jax.lax.stop_gradient(n)
    
    rho_64 = to_fp64(rho)
    T_64 = to_fp64(T)
    n_64 = to_fp64(n)
    
    # Debug shapes
    # jax.debug.print("get_sound_speed: rho={}, T={}, n shape={}", rho_64, T_64, n_64.shape)

    # --- Unit Normalization (The Fix) ---
    rho_kg_m3 = _rho_to_kg_per_m3(rho_64)

    m_kg, mw_avg_kg_per_mol, n_tot = (
        _mixture_mass_and_mw_avg_kg_per_mol(n, atom_vec, atomic_masses)
    )

    V_total_m3 = m_kg / rho_kg_m3  # m^3 (total volume for the given 'recipe unit')

    if _DEBUG_EOS_UNITS:
        V_molar_cm3 = (V_total_m3 / (n_tot + 1e-300)) * 1e6
        jax.debug.print(
            "EOS-UNITS: rho_in={:.6g}, rho_SI={:.6g} kg/m3, n_tot={:.6g} mol, "
            "m={:.6g} kg, mw_avg={:.6g} kg/mol, V_total={:.6g} m3, V_molar={:.6g} cm3/mol",
            rho, rho_kg_m3, n_tot, m_kg, mw_avg_kg_per_mol, V_total_m3, V_molar_cm3
        )
    
    # 定义 A(V_cm3, T) - Adapter for compute_total_helmholtz_energy
    # Ensure compute_total_helmholtz_energy receives consistent units.
    # It takes "V_total".
    
    def A_of_VT(Vc, Tc):
        # Vc is in cm3
        res = compute_total_helmholtz_energy(
            n_64, Vc, Tc, 
            coeffs_low, coeffs_high, 
            *eos_params, 
            mw_avg=mw_avg_kg_per_mol # Pass the CORRECT MW
        )
        # jax.debug.print("A_val shape/val: {}", res)
        return jnp.sum(res) # Force scalar just in case, but this implies bug in EOS if vector.
    
    V_cm3 = V_total_m3 * 1e6
    rho_si = rho_kg_m3

    # 一阶导
    dA_dV = jax.grad(A_of_VT, argnums=0)
    dA_dT = jax.grad(A_of_VT, argnums=1)
    
    # 二阶导
    d2A_dV2 = jax.grad(lambda v, t: dA_dV(v, t), argnums=0)
    d2A_dVdT = jax.grad(lambda v, t: dA_dV(v, t), argnums=1)
    d2A_dT2 = jax.grad(lambda v, t: dA_dT(v, t), argnums=1)
    
    # 计算导数值
    A_V = dA_dV(V_cm3, T_64)      # J/cm3
    A_VV = d2A_dV2(V_cm3, T_64)   # J/cm6
    A_VT = d2A_dVdT(V_cm3, T_64)  # J/(cm3 K)
    A_TT = d2A_dT2(V_cm3, T_64)   # J/K2
    
    # 转 SI 压强与导数
    # P = -dA/dV (J/cm3 = MPa = 1e6 Pa)
    P = -A_V * 1e6                 # Pa
    P_T = -A_VT * 1e6              # Pa/K
    P_V_T = -A_VV * 1e12           # Pa/m3 (cm6 -> m6 is 1e-12, denominator)
    
    # Cv_total = -T * d2A/dT2 (J/K)
    Cv_total = -T_64 * A_TT
    
    # 数值保护: Cv 必须正
    # Use smooth_floor but need to import or ensure it is available (imported from eos at top)
    Cv_min = 1e-6
    Cv_total = smooth_floor(Cv_total, Cv_min, 1e-6)
    cv_mass = Cv_total / m_kg      # J/kg/K (consistent with mass)
    
    # 热力学公式: a^2 = (dP/drho)_s
    # (dP/drho)_T = (dP/dV)_T * (dV/drho) = (dP/dV)_T * (-V/rho)
    dP_drho_T = P_V_T * (-V_total_m3 / rho_si) # Pa / (kg/m3) = m2/s2
    
    # 绝热声速平方
    # a^2 = (dP/drho)_T + (T / (rho^2 * cv)) * (dP/dT)^2
    a2 = dP_drho_T + (T_64 / (rho_si**2 * cv_mass)) * (P_T**2)
    
    return to_fp32(jnp.sqrt(jnp.maximum(a2, 1.0)))

@jit
def get_internal_energy_pt(rho, T, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params):
    rho_64 = to_fp64(rho)
    T_64 = to_fp64(T)
    mw_g_per_mol = jnp.dot(atom_vec, atomic_masses)
    V_molar = mw_g_per_mol / (rho_64 + 1e-10)
    
    n_eq = solve_equilibrium(atom_vec, V_molar, T_64, A_matrix, coeffs_low, coeffs_high, eos_params)
    u_mol = compute_internal_energy_jcz3(n_eq, V_molar, T_64, coeffs_low, coeffs_high, *eos_params, mw_g_per_mol)
    
    return to_fp32(u_mol / (mw_g_per_mol / 1000.0))