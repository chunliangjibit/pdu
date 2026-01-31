# tests/diagnose_thermo_baseline.py
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pdu.physics.eos import compute_total_helmholtz_energy, _get_thermo_vec
from pdu.data.products import load_products
from pdu.utils.precision import R_GAS, to_fp64
import json
from pathlib import Path

def diagnose():
    print("=== PDU Thermo Baseline Diagnosis ===")
    
    # 1. Setup sample state (Low density - Ideal Gas Limit)
    T = 4000.0
    V_total = 1e6 # 1 m3 - low but safe density
    coeffs_low = jnp.zeros((7, 9))
    coeffs_high = jnp.zeros((7, 9))
    
    # Load H2O NASA coeffs
    db = load_products()
    h2o = db['H2O']
    # Pad to 9
    ch = jnp.concatenate([jnp.zeros(2), h2o.coeffs_high[:7]])
    cl = jnp.concatenate([jnp.zeros(2), h2o.coeffs_low[:7]])
    
    # Species: H2O only
    n = jnp.array([1.0, 0, 0, 0, 0, 0, 0])
    solid_mask = jnp.array([0., 0., 0., 0., 0., 0., 0.])
    solid_v0 = jnp.array([0., 0., 0., 0., 0., 0., 0.])
    
    # Load standard JCZ3 params (but they should have near 0 impact at V=1e10)
    # Mapping to correct params
    eps_v = jnp.array([333.0, 0, 0, 0, 0, 0, 0])
    r_v = jnp.array([3.27, 0, 0, 0, 0, 0, 0])
    alpha_v = jnp.array([13.0, 0, 0, 0, 0, 0, 0])
    lam_v = jnp.array([1500.0, 0, 0, 0, 0, 0, 0])
    
    params = (eps_v, r_v, alpha_v, lam_v, solid_mask, 5.3) # solid_v0 as scalar
    
    # --- Test 1: Ideal Gas Internal Energy via NASA vs Helmholtz ---
    # NASA Direct (u = h - RT)
    u_nasa_vec, s_nasa_vec = _get_thermo_vec(jnp.stack([cl]*7), jnp.stack([ch]*7), T)
    u_nasa_h2o = u_nasa_vec[0]
    
    # Helmholtz Derived Components Grad Check
    def check_grad(name, func_of_t):
        val = func_of_t(T)
        grad = jax.grad(func_of_t)(T)
        print(f"{name:10} | val: {val:12.3f} | grad: {grad:12.3f}")

    print("\n--- Component-wise Gradient Analysis ---")
    
    # Needs to be wrapped to pass all params
    def run_part(temp, mode):
        # We need to reach into compute_total_helmholtz_energy logic
        # OR just mock the call with internal parts
        from pdu.physics.eos import compute_covolume, compute_mixed_matrices_dynamic, _get_thermo_vec, smooth_switch, R_GAS
        
        eps_mat, r_mat, alpha_mat = compute_mixed_matrices_dynamic(temp, eps_v, r_v, alpha_v, lam_v)
        u_vec, s0_vec = _get_thermo_vec(jnp.stack([cl]*7), jnp.stack([ch]*7), temp)
        
        # A_gas_ideal calculation
        V_gas_m3 = V_total * 1e-6
        n_gas = n # H2O only
        n_gas_safe = jnp.maximum(n_gas, 1e-15)
        A_gas_ideal = jnp.sum(n_gas * (u_vec - temp * s0_vec)) + R_GAS * temp * jnp.sum(jnp.where(n_gas > 1e-18, n_gas * jnp.log((n_gas_safe * R_GAS * temp) / (V_gas_m3 * 1e5)), 0.0))
        
        if mode == 'ideal': return A_gas_ideal
        
        B_total_gas = compute_covolume(n_gas, r_mat, temp, eps_mat, alpha_mat, 0.0)
        if mode == 'B': return B_total_gas
        eta_raw = B_total_gas / (4.0 * V_total)
        
        ETA_CAP = 0.95
        ETA_CAP_W = 0.005
        from pdu.physics.eos import smooth_cap, smooth_floor
        eta_cs = smooth_cap(eta_raw, ETA_CAP, ETA_CAP_W)
        one_minus = smooth_floor(1.0 - eta_cs, 1e-6, 1e-6)
        f_cs = (4.0 * eta_cs - 3.0 * eta_cs * eta_cs) / (one_minus * one_minus)
        A_cs = 1.0 * R_GAS * temp * f_cs
        
        # A_bar check
        OVER_W = 0.02
        over = jax.nn.softplus((eta_raw - ETA_CAP) / OVER_W) * OVER_W
        K_ETA = 100.0
        A_bar = 1.0 * R_GAS * temp * K_ETA * (over / OVER_W) ** 2
        
        if mode == 'hs_cs': return A_cs
        if mode == 'hs_bar': return A_bar
        if mode == 'hs': return A_cs + A_bar
        
        U_attr = - ((2.0 * jnp.pi / 3.0) * (6.02214076e23**2) * 1.380649e-23 * 1e-24 * jnp.sum(jnp.outer(n_gas, n_gas) * eps_mat * r_mat**3)) / V_total

        if mode == 'attr': return U_attr
        return 0.0

    print("\n--- Ratio Assembly Analysis ---")
    def check_ratio_assembly(temp):
        from pdu.physics.eos import compute_mixed_matrices_dynamic, smooth_floor, smooth_cap
        eps_mat, r_mat, alpha_mat = compute_mixed_matrices_dynamic(temp, eps_v, r_v, alpha_v, lam_v)
        T_safe = smooth_floor(temp, 100.0, 10.0)
        eps_safe = jnp.maximum(eps_mat, 1.0)
        T_star = T_safe / eps_safe
        term = ((alpha_mat - 6.0) / 6.0) * T_star
        log_term = jnp.log(jnp.maximum(term, 1e-10))
        # Culprit hunt:
        inv_alpha = 1.0 / jnp.maximum(alpha_mat, 1.0)
        ratio_raw = 1.0 - inv_alpha * log_term
        return jnp.sum(ratio_raw)

    print(f"Ratio Assembly Grad: {jax.grad(check_ratio_assembly)(T)}")

    print(f"Ratio Assembly Grad: {jax.grad(check_ratio_assembly)(T)}")
    
    print("\n--- Final Helmholtz vs NASA Baseline Comparison ---")
    def get_U_final(temp):
        A = compute_total_helmholtz_energy(n, V_total, temp, jnp.stack([cl]*7), jnp.stack([ch]*7), *params, mw_avg=18.015)
        grad_T = jax.grad(compute_total_helmholtz_energy, argnums=2)(n, V_total, temp, jnp.stack([cl]*7), jnp.stack([ch]*7), *params, mw_avg=18.015)
        return A - temp * grad_T

    u_helm = get_U_final(T)
    print(f"T:         {T} K")
    print(f"NASA U:    {u_nasa_h2o:.3f} J/mol")
    print(f"Helm U:    {u_helm:.3f} J/mol")
    print(f"Diff:      {u_helm - u_nasa_h2o:.3f} J/mol")
    print(f"Diff/RT:   {(u_helm - u_nasa_h2o)/(R_GAS*T):.6f}")
    
    # --- Test 2: Pressure Check ---
    def get_P(vol):
        # returns P in J/cm3
        A_grad_V = jax.grad(compute_total_helmholtz_energy, argnums=1)(n, vol, T, jnp.stack([cl]*7), jnp.stack([ch]*7), *params, mw_avg=18.015)
        return -A_grad_V * 1e6 # Pa

    P_calc = get_P(V_total)
    P_ideal = (1.0 * R_GAS * T) / (V_total * 1e-6)
    print(f"\nIdeal Pressure (RT/V): {P_ideal:.3f} Pa")
    print(f"Calculated Pressure:    {P_calc:.3f} Pa")
    print(f"Pressure Ratio:         {P_calc/P_ideal:.6f}")
    
if __name__ == "__main__":
    diagnose()
