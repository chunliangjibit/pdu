"""
Enhanced Differentiable CJ Solver with Isentrope Calculation

Returns: (D_cj, P_cj, T_cj, isentrope_data) for JWL fitting
"""

import jax
import jax.numpy as jnp
from pdu.core.equilibrium import solve_equilibrium
from pdu.physics.eos import compute_pressure_jcz3, compute_internal_energy_jcz3, compute_entropy_consistent
# 【修复】使用自洽熵替代NASA理想气体熵
from typing import Tuple, Optional, Dict, List


from functools import partial

R_GAS = 8.314  # J/(mol·K)

# 【废弃】旧的理想气体熵计算已移除
# V6版本手动实现的 compute_entropy_jcz3 忽略了剩余熵 S_res
# 现在使用 pdu.physics.eos.compute_entropy_consistent (JAX AD 自动微分)
# 保证了 S, P, U 的热力学自洽性


def predict_cj_with_isentrope(
    eps_vec: jnp.ndarray,
    r_star_vec: jnp.ndarray,
    alpha_vec: jnp.ndarray,
    lambda_vec: jnp.ndarray,
    V0: float,
    E0: float,
    atom_vec: jnp.ndarray,
    coeffs_all: jnp.ndarray,
    A_matrix: jnp.ndarray,
    atomic_masses: jnp.ndarray,
    n_isentrope_points: int = 30,
    solid_mask: Optional[jnp.ndarray] = None,
    solid_v0: Optional[jnp.ndarray] = None,
    n_fixed_inert: Optional[float] = 0.0,
    v0_fixed_inert: Optional[float] = 10.0,
    e_fixed_inert: Optional[float] = 0.0,
    r_star_rho_corr: float = 0.0
) -> tuple:
    """
    Enhanced CJ solver (V8 Multi-Phase Version + V9 Partial Reaction).
    """
    
    # Calculate Molar Mass
    Mw_g = jnp.dot(atom_vec, atomic_masses)
    Mw_kg = Mw_g / 1000.0
    rho0 = Mw_kg / (V0 * 1e-6)
    
    # Eos params tuple for equilibrium
    # V10: Extended tuple with Ree lambda parameter
    eos_params = (eps_vec, r_star_vec, alpha_vec, lambda_vec, solid_mask, solid_v0, 
                  n_fixed_inert, v0_fixed_inert, e_fixed_inert, r_star_rho_corr)
    
    # =========================================================================
    # Step 1: Find CJ Point
    # =========================================================================
    
    def scan_hugoniot(carry, vr):
        V_test = V0 * vr
        
        # Bisection for T at this V
        T_min = 300.0
        T_max = 12000.0
        
        def bisect_step(i, bounds):
            t_lo, t_hi = bounds
            t_mid = (t_lo + t_hi) / 2.0
            
            
            n_eq = solve_equilibrium(atom_vec, V_test, t_mid, A_matrix, coeffs_all, eos_params)
            
            E = compute_internal_energy_jcz3(n_eq, V_test, t_mid, coeffs_all, eps_vec, r_star_vec, alpha_vec, lambda_vec, solid_mask, solid_v0, n_fixed_inert, v0_fixed_inert, e_fixed_inert, r_star_rho_corr)
            P = compute_pressure_jcz3(n_eq, V_test, t_mid, coeffs_all, eps_vec, r_star_vec, alpha_vec, lambda_vec, solid_mask, solid_v0, n_fixed_inert, v0_fixed_inert, e_fixed_inert, r_star_rho_corr)
            
            dV = (V0 - V_test) * 1e-6
            hug_err = (E - E0) - 0.5 * P * dV
            
            new_lo = jnp.where(hug_err > 0, t_lo, t_mid)
            new_hi = jnp.where(hug_err > 0, t_mid, t_hi)
            return (new_lo, new_hi)
        
        t_bounds = jax.lax.fori_loop(0, 18, bisect_step, (T_min, T_max))
        T_hug = (t_bounds[0] + t_bounds[1]) / 2.0
        
        
        n_eq = solve_equilibrium(atom_vec, V_test, T_hug, A_matrix, coeffs_all, eos_params)
        P_pa = compute_pressure_jcz3(n_eq, V_test, T_hug, coeffs_all, eps_vec, r_star_vec, alpha_vec, lambda_vec, solid_mask, solid_v0, n_fixed_inert, v0_fixed_inert, e_fixed_inert, r_star_rho_corr)
        
        D_sq = P_pa / jnp.maximum(rho0 * (1.0 - vr), 1e-6)
        D = jnp.sqrt(jnp.maximum(D_sq, 0.0))
        
        return carry, (D, P_pa, T_hug, V_test, n_eq)
    
    v_ratios = jnp.linspace(0.5, 0.85, 20)
    _, hug_data = jax.lax.scan(scan_hugoniot, None, v_ratios)
    
    D_grid, P_grid, T_grid, V_grid, n_grid = hug_data
    
    # Find CJ point (minimum D)
    cj_idx = jnp.argmin(D_grid)
    D_cj = D_grid[cj_idx]
    P_cj = P_grid[cj_idx] / 1e9  # Convert to GPa
    T_cj = T_grid[cj_idx]
    V_cj = V_grid[cj_idx]
    n_cj = n_grid[cj_idx]
    
    # =========================================================================
    
    S_cj = compute_entropy_consistent(n_cj, V_cj, T_cj, coeffs_all, eps_vec, r_star_vec, alpha_vec, lambda_vec, solid_mask, solid_v0, n_fixed_inert, v0_fixed_inert, e_fixed_inert, r_star_rho_corr)
    
    def scan_isentrope(carry, v_ratio):
        V_iso = V0 * v_ratio
        
        # Find T where S(V, T) = S_cj (isoentropic expansion)
        
        # Find T where S(V, T) = S_cj (isoentropic expansion)
        T_low = 300.0
        T_high = T_cj + 2000.0
        
        def entropy_bisect(i, bounds):
            t_lo, t_hi = bounds
            t_mid = (t_lo + t_hi) / 2.0
            
            n_iso = solve_equilibrium(atom_vec, V_iso, t_mid, A_matrix, coeffs_all, eos_params)
            
            S_curr = compute_entropy_consistent(n_iso, V_iso, t_mid, coeffs_all, eps_vec, r_star_vec, alpha_vec, lambda_vec, solid_mask, solid_v0, n_fixed_inert, v0_fixed_inert, e_fixed_inert, r_star_rho_corr)
            
            # S decreases with T (typically)? Actually S increases with T.
            # If S_curr > S_cj, T is too high.
            new_lo = jnp.where(S_curr > S_cj, t_lo, t_mid)
            new_hi = jnp.where(S_curr > S_cj, t_mid, t_hi)
            return (new_lo, new_hi)
        
        t_bounds_iso = jax.lax.fori_loop(0, 18, entropy_bisect, (T_low, T_high))
        T_iso = (t_bounds_iso[0] + t_bounds_iso[1]) / 2.0
        
        # Compute pressure at this isentrope point
        n_iso = solve_equilibrium(atom_vec, V_iso, T_iso, A_matrix, coeffs_all, eos_params)
        P_pa = compute_pressure_jcz3(n_iso, V_iso, T_iso, coeffs_all, eps_vec, r_star_vec, alpha_vec, lambda_vec, solid_mask, solid_v0, n_fixed_inert, v0_fixed_inert, e_fixed_inert, r_star_rho_corr)
        
        return carry, (V_iso, P_pa / 1e9)
    
    # Scan expansion from V_cj to 8.0*V0 (Standard JWL range)
    # We use a fixed size for JIT, but here n_isentrope_points is passed from top
    iso_v_ratios = jnp.linspace(V_cj/V0, 8.0, n_isentrope_points)
    _, iso_data = jax.lax.scan(scan_isentrope, None, iso_v_ratios)
    
    return D_cj, P_cj, T_cj, V_cj, iso_data[0], iso_data[1], n_cj


# Simplified wrapper for backward compatibility
@jax.jit
def predict_D_cj(
    eos_params_flat,
    species_indices,
    fixed_params,
    epsilon_matrix,
    r_star_matrix,
    alpha_matrix,
    V0,
    E0,
    atom_vec,
    coeffs_all,
    A_matrix,
    atomic_masses
) -> float:
    """Legacy wrapper - returns only D_cj"""
    D_cj, _, _, _, _, _ = predict_cj_with_isentrope(
        epsilon_matrix, r_star_matrix, alpha_matrix,
        V0, E0, atom_vec, coeffs_all, A_matrix, atomic_masses,
        n_isentrope_points=10  # Reduced for speed
    )
    return D_cj
