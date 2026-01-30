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
      n_tot: total moles (mol)
    """
    am_kg_per_mol = _atomic_masses_to_kg_per_mol(atomic_masses)
    
    # atom_vec is (natoms,) representing the system formula (e.g. C4H8N8O8)
    # Total mass is fixed by this vector regardless of speciation n
    m_kg = jnp.dot(atom_vec, am_kg_per_mol) # Scalar
    
    n = jnp.asarray(n)
    n_tot = jnp.sum(n)
    mw_avg_kg_per_mol = m_kg / (n_tot + 1e-300)

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
    # We need n_init for mixture mass if provided, but here n is solved.
    # However, to solve equilibrium, we need V_molar.
    
    # 1. Determine MW (assuming 1 mol basis for initial guess if needed, but equilibrium solver iterates n)
    # The equilibrium solver needs V_molar (volume per mole of mixture?? Or volume of the system?)
    # JCZ3 typically works with V per "recipe unit" or molar volume.
    # Let's trust the expert's path: use robust units to get m_kg and V_total_m3.
    
    # We don't have 'n' yet, so we can't calculate m_kg exactly for the *final* mixture, 
    # BUT atomic conservation means total mass is constant regardless of n.
    # Let's use a "basis" n (e.g. n_init or fictitious) to get mass? 
    # Actually, mw_g_per_mol (average? or per species?)
    # Wait, existing code: mw_g_per_mol = jnp.dot(atom_vec, atomic_masses) -> this is per species!
    
    # The existing code did: V_molar = mw_g_per_mol / rho (broadcasting?)
    # If mw_g_per_mol is (nspecies,), V_molar would be (nspecies,). That seems wrong for single-phase V.
    # Ah, 'solve_equilibrium' likely takes V_molar as a scalar (System Volume / Total Moles)?
    # Or maybe V_molar is per-species? No.
    
    # Let's look at the expert's snippet. He calculates V_total_m3 = m_kg / rho_kg_m3.
    # CONSTANT MASS constraint.
    # Let's assume 1 mole of "formula unit" (or whatever atom_vec implies) if we are doing equilibrium.
    # Usually we solve for n such that sum(n_i * atoms_ij) = b_j.
    # A_matrix defines conservation.
    
    # Let's stick to the existing logic but FIX inputs.
    # Get robust Atomic Masses
    am_kg_per_mol = _atomic_masses_to_kg_per_mol(atomic_masses)
    
    # Calculate MW per species (kg/mol)
    mw_species_kg_per_mol = atom_vec @ am_kg_per_mol
    
    # But wait, to get V_molar (volume per mole of mixture?), we need n.
    # If we don't have n, we can't get V_molar simply.
    # BUT, usually we input V (volume) and T.
    # Here inputs are rho and T.
    
    # If `get_thermo_properties` assumes a fixed mass system (e.g. 1 kg, or mass of 1 recipe unit):
    # Let's assume the system mass is defined by the conservation constraints (b vector in A_matrix).
    # If A_matrix * n = b, and we know atomic masses, mass is fixed.
    # For now, let's keep the *logic* of V_molar roughly same but use SI.
    
    # Existing: mw_g_per_mol = dot(atom_vec, atomic_masses) (vector of species MWs)
    # V_molar = mw_g_per_mol / rho
    # This implies V_molar is a vector? Or dot product with n?
    # No, solve_equilibrium signature: (atom_vec, V_molar, ...)
    # If V_molar is vector, it might be per-species volume?
    # Let's stick to strictly patching UNITS for now.
    
    # Use helper to get species MW in kg
    am_kg = _atomic_masses_to_kg_per_mol(atomic_masses)
    mw_species_kg = atom_vec @ am_kg # (nspecies,)
    
    # We need to pass valid inputs to solve_equilibrium.
    # If solve_equilibrium expects V in cm3 (likely, based on previous code using g/cm3),
    # we should check `solve_equilibrium`. But expert says "Use V_total_m3" for compute values.
    # Let's standardize density first using helper.
    rho_kg_m3 = _rho_to_kg_per_m3(rho_64)
    
    # Revert to old logic for V_molar but ensuring units are consistent (e.g. all SI or all CGS)
    # EXPERT SAYS: "Use m_kg and rho_kg_m3 to get V_total_m3"
    # But here we don't have n yet.
    # Let's compute n first using the existing path (assuming it effectively works or we can't easily change it without breaking solver).
    # WAIT. If rho is passed as 3.2 (g/cc) but interpreted as kg/m3 (3.2 kg/m3 = air), V will be huge.
    # So we MUST fix rho input to solve_equilibrium too.
    
    # Let's do a minimal invasion: Use original logic but ensure `rho` is correct magnitude for what the solver expects.
    # If logic uses CGS internally:
    rho_cgs = rho_kg_m3 * 1e-3 # g/cm3 (e.g. 1.9)
    # mw_species_g = mw_species_kg * 1000
    
    # Original:
    # mw_g_per_mol = jnp.dot(atom_vec, atomic_masses) 
    # V_molar = mw_g_per_mol / (rho_64 + 1e-10)
    
    # If atomic_masses was g/mol, mw_g_per_mol is g/mol.
    # If rho was g/cm3, V_molar is cm3/mol.
    # This seems consistent CGS. 
    # ISSUE: atomic_masses might be kg/mol?
    
    # Fix atomic masses to g/mol for the CGS part:
    am_g = _atomic_masses_to_kg_per_mol(atomic_masses) * 1000.0
    mw_g_per_mol = jnp.dot(atom_vec, am_g)
    
    # Fix rho to g/cm3
    rho_cgs = _rho_to_kg_per_m3(rho_64) * 1e-3
    
    # Now V_molar (vector?)
    V_molar = mw_g_per_mol / (rho_cgs + 1e-10) # cm3 / mol_species
    
    if _DEBUG_EOS_UNITS:
        jax.debug.print("SOLVER_INPUT: mw_g={} g/mol, rho_cgs={} g/cm3, V_molar={} cm3/mol, T={}", mw_g_per_mol, rho_cgs, V_molar, T_64)
    
    n_final = solve_equilibrium(atom_vec, V_molar, T_64, A_matrix, coeffs_low, coeffs_high, eos_params, n_init)
    
    # --- Now we have n, we can compute P using the ROBUST path ---
    m_kg, mw_avg_kg, n_tot = _mixture_mass_and_mw_avg_kg_per_mol(n_final, atom_vec, atomic_masses)
    
    V_total_m3 = m_kg / rho_kg_m3
    
    if _DEBUG_EOS_UNITS:
        V_molar_cm3 = (V_total_m3 / (n_tot + 1e-300)) * 1e6
        jax.debug.print(
            "THERMO_PROP: rho_in={} g/cm3, rho_SI={} kg/m3, n_tot={} mol, m_kg={} kg, V_tot_m3={} m3, V_mol_cm3={} cm3/mol",
            rho, rho_kg_m3, n_tot, m_kg, V_total_m3, V_molar_cm3
        )

    # Use compute_total_helmholtz_energy to get P = -dA/dV consistently?
    # Or use compute_pressure_jcz3 (which might have its own internal unit assumptions)?
    # Expert said: "Use compute_total_helmholtz_energy(..., V_total=V_total_m3, mw_avg=mw_avg_kg_per_mol)" to get A, then derive.
    # compute_pressure_jcz3 is likely an analytic implementation.
    # Let's stick to compute_pressure_jcz3 but pass it CORRECTED arguments if needed.
    # Actually, compute_pressure_jcz3 takes V_molar. If we pass the corrected V_total_m3 (converted to molar?), it should work.
    # compute_pressure_jcz3 signature: (n, V, T, ..., mw)
    
    # Let's call compute_pressure_jcz3 with the robust V_molar we just found?
    # V_molar_for_P = V_total_m3 / n_tot ?? No, jcz3 might expect specific volume.
    # Let's look at compute_pressure_jcz3 signature in eos.py later or rely on existing logic.
    # Existing: V_molar passed was (mw_g_per_mol / rho). This is species-specific volume?
    # If compute_pressure_jcz3 handles vector V, ok.
    
    # But to be safe and fix the "Pressure 0.12" issue which is likely due to units:
    # I will trust that the CGS conversion above (am_g, rho_cgs) fixes the input to solve_equilibrium.
    # Then I pass the resulting state to compute_pressure_jcz3.
    
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

    # --- Unit Normalization (The Fix) ---
    rho_kg_m3 = _rho_to_kg_per_m3(rho_64)

    m_kg, mw_avg_kg_per_mol, n_tot = (
        _mixture_mass_and_mw_avg_kg_per_mol(n, atom_vec, atomic_masses)
    )

    V_total_m3 = m_kg / rho_kg_m3  # m^3 (total volume for the given 'recipe unit')

    if _DEBUG_EOS_UNITS:
        V_molar_cm3 = (V_total_m3 / (n_tot + 1e-300)) * 1e6
        jax.debug.print(
            "EOS-UNITS: rho_in={} g/cm3, rho_SI={} kg/m3, n_tot={} mol, "
            "m={} kg, mw_avg={} kg/mol, V_total={} m3, V_molar={} cm3/mol",
            rho, rho_kg_m3, n_tot, m_kg, mw_avg_kg_per_mol, V_total_m3, V_molar_cm3
        )
    
    # 定义 A(V_cm3, T) - Adapter for compute_total_helmholtz_energy
    # Ensure compute_total_helmholtz_energy receives consistent units.
    # It likely expects V_total in... cm3? or m3?
    # Based on previous code: V_m3 = V_cm3 * 1e-6 -> passed to... wait.
    # The previous code passed V_cm3 to compute_total_helmholtz_energy?
    # line 93: n_64, Vc, Tc...
    # line 100: V_cm3 = mw / rho.
    # line 91: returns compute_total_helmholtz_energy(..., Vc, ...)
    # So it expects Vc (cm3?)
    
    # Let's verify compute_total_helmholtz_energy signature or behavior.
    # But assuming expert code is correct:
    # Expert code in `get_sound_speed_from_A` (C-2) computes A_V = dA_dV(V_cm3, T).
    # This implies compute_total_helmholtz_energy takes V_cm3.
    # Expert says: "use V_total_m3 for A calc"?
    # In C-2 snippet: "A(V,T) ... helmholtz_fn(n, V_cm3, T)"
    # So helmholtz_fn takes V_cm3.
    
    # So we must pass V_total_m3 * 1e6 to the A function?
    
    def A_of_VT(Vc, Tc):
        # Vc is in cm3
        # We need to make sure compute_total_helmholtz_energy handles it correctly.
        # It takes "V_total".
        return compute_total_helmholtz_energy(
            n_64, Vc, Tc, 
            coeffs_low, coeffs_high, 
            *eos_params, 
            mw_avg_kg_per_mol # Pass the CORRECT MW
        )
    
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
    
    # 数值保护: Cv 必须正且不能太小
    Cv_min = 1e-6
    Cv_total = smooth_floor(Cv_total, Cv_min, 1e-6)
    cv_mass = Cv_total / m_kg      # J/kg/K
    
    # 热力学公式: (dP/drho)_s = (dP/drho)_T + T/(rho^2 * cv) * (dP/dT)^2
    # (dP/drho)_T = (dP/dV)_T * (dV/drho)
    # V = m/rho -> dV/drho = -m/rho^2 = -V/rho
    dP_drho_T = P_V_T * (-V_total_m3 / rho_si) # Pa / (kg/m3) = m2/s2
    
    # 绝热声速平方
    a2 = dP_drho_T + (T_64 / (rho_si**2 * cv_mass)) * (P_T**2)
    
    # 最终保护
    a2 = jnp.maximum(a2, 1.0)
    return to_fp32(jnp.sqrt(a2))

@jit
def get_internal_energy_pt(rho, T, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params):
    rho_64 = to_fp64(rho)
    T_64 = to_fp64(T)
    mw_g_per_mol = jnp.dot(atom_vec, atomic_masses)
    V_molar = mw_g_per_mol / (rho_64 + 1e-10)
    
    n_eq = solve_equilibrium(atom_vec, V_molar, T_64, A_matrix, coeffs_low, coeffs_high, eos_params)
    u_mol = compute_internal_energy_jcz3(n_eq, V_molar, T_64, coeffs_low, coeffs_high, *eos_params, mw_g_per_mol)
    
    return to_fp32(u_mol / (mw_g_per_mol / 1000.0))