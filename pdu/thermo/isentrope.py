import jax
import jax.numpy as jnp
from jax import jit, lax
from pdu.thermo.implicit_eos import get_thermo_properties, compute_entropy_consistent
from pdu.utils.precision import to_fp64, to_fp32

@jit
def solve_T_isentropic(S_target, V_target, T_guess, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params):
    """
    Solve for T such that S(V_target, T) = S_target using Newton's method.
    Assumes Chemical Equilibrium at (V_target, T).
    """
    def entropy_residual(T_curr):
        # 1. Solve Equilibrium at (V, T)
        P, n_eq = get_thermo_properties(
            jnp.array(1.0), T_curr, # rho is dummy here, get_thermo handles n by solving V,T
            atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params
        )
        # Note: get_thermo_properties takes rho, T. 
        # But we have V_total. We need rho as input. 
        # But rho depends on Mass, which depends on n?
        # Actually n_eq implies Mass...
        # Wait, get_thermo_properties signature is (rho, T, ...).
        # We need to adapt this.
        pass
    
    # We need a direct way to compute S(n_eq(V,T), V, T).
    # implicit_eos.py has get_thermo_properties(rho, T).
    # We can compute rho = mass / V_target. (Approx mass first?)
    # Mass is invariant if atom_vec is fixed!
    # m_kg = dot(atom_vec, atomic_masses). 
    # So rho IS known given V_target.
    
    am_kg = jnp.where(jnp.max(atomic_masses)>0.5, atomic_masses*1e-3, atomic_masses)
    m_kg_total = jnp.dot(atom_vec, am_kg)
    rho_target = m_kg_total / (V_target + 1e-30) # V in m3
    
    # Check units: V_target should be m3? or cm3?
    # JWL fitting usually works in Relative Volume or specific units.
    # Let's assume V_target is m3 (SI).
    
    def scalar_newton_step(T_val):
        # S(T)
        # We need gradients dS/dT.
        # compute_entropy_consistent gives S.
        
        # We can use AD.
        def get_s(t):
            # Recalculate rho? No rho is fixed for fixed V.
            # But get_thermo_properties solves equilibrium internally.
            P, n = get_thermo_properties(rho_target, t, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
            
            # Now compute S.
            # We need n_eq to compute S consistent?
            # compute_entropy_consistent takes n, V, T...
            # But n changes with T. 
            # If we assume Equilibrium, dS/dT includes dn/dT term.
            # get_thermo_properties returns n_eq.
            
            # Wait, implicit_eos doesn't expose a clean "get S(rho, T)" function that handles equilibrium.
            # It only has compute_entropy_consistent(n, ...).
            # We should wrap it.
            
            # But get_thermo_properties is @custom_jvp. It might not differentiate n w.r.t T correctly for S?
            # Actually get_thermo_properties returns (P, n). P is diffable. n is diffable.
            
            # Unit conversion for compute_entropy_consistent?
            # It expects V_total, T, ...
            
            # Let's trust n returned by get_thermo_properties.
            S = compute_entropy_consistent(
                n, V_target, t, coeffs_low, coeffs_high, 
                *eos_params,
                jnp.dot(atom_vec, atomic_masses) # mw_avg placeholder?
            )
            return S
        
        S_val = get_s(T_val)
        dS_dT = jax.grad(get_s)(T_val)
        
        return S_val, dS_dT

    # Newton Loop
    T = T_guess
    for i in range(15):
        S_val, dS_dT = scalar_newton_step(T)
        R = S_val - S_target
        dT = -R / (dS_dT + 1e-10)
        
        # Damping
        dT = jnp.clip(dT, -0.2*T, 0.2*T)
        T = T + dT
            
    return T

from functools import partial

@partial(jit, static_argnames=['n_points'])
def generate_isentrope(rho_cj, T_cj, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params, v_max_rel=10.0, n_points=20):
    """
    Generate isentrope from V_cj to V_max_rel * V0.
    """
    # 1. Get CJ State Entropy
    P_cj, n_cj = get_thermo_properties(rho_cj, T_cj, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
    
    am_kg = jnp.where(jnp.max(atomic_masses)>0.5, atomic_masses*1e-3, atomic_masses)
    m_kg = jnp.dot(atom_vec, am_kg)
    V_cj = m_kg / rho_cj # m3
    
    # Note: implicit_eos.py compute_entropy_consistent signature:
    # (n, V_total, T, coeffs_low, coeffs_high, eps_vec, ..., mw_avg)
    # We need to construct *args equivalent to *eos_params
    
    # Wait, compute_entropy_consistent uses *args.
    # We must unpack eos_params exactly as implicit_eos does.
    # In implicit_eos.py:
    # compute_pressure_jcz3(..., eos_params[0], eos_params[1], ... eos_params[9], mw_avg)
    
    # We need to match this unpacking.
    
    mw_avg_dummy = 1.0 
    # S_cj calculation
    S_cj = compute_entropy_consistent(
        n_cj, V_cj, T_cj, coeffs_low, coeffs_high,
        eos_params[0], eos_params[1], eos_params[2], eos_params[3],
        eos_params[4], eos_params[5], 
        eos_params[6], eos_params[7], eos_params[8],
        eos_params[9],
        mw_avg_dummy
    )
    
    # 2. Volumes
    # V0 = V_cj? No, V0 is initial explosive specific volume.
    # rho0 = 1.891 (HMX). 
    # But usually JWL is fit relative to V0 = 1/rho0. 
    # But here we are just generating P-V check points.
    
    # Let's generate relative to V_cj.
    # V_cj is usually V_rel ~ 0.75.
    # We want V up to 7-10 times V_cj.
    
    V_factors = jnp.linspace(1.0, v_max_rel, n_points)
    V_array = V_cj * V_factors
    
    def verify_point(V_curr, T_guess):
        T_new = solve_T_isentropic(S_cj, V_curr, T_guess, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
        
        am_kg = jnp.where(jnp.max(atomic_masses)>0.5, atomic_masses*1e-3, atomic_masses)
        m_kg = jnp.dot(atom_vec, am_kg)
        rho = m_kg / V_curr
        P, n = get_thermo_properties(rho, T_new, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
        return P, T_new
    
    # Scan
    P_list = []
    T_list = []
    T_curr = T_cj
    
    # Use Lax scan for efficiency? Or Python loop for debug?
    # Python loop is fine for 20 points.
    
    # We CANNOT use python list in JIT.
    # Must use lax.scan.
    
    def scan_fn(carry, V):
        T_prev = carry
        P, T_next = verify_point(V, T_prev)
        return T_next, (P, T_next)
        
    _, (P_vec, T_vec) = lax.scan(scan_fn, T_cj, V_array)
    
    return V_array, P_vec, T_vec
