# pdu/solver/jump_conditions.py
import jax
import jax.numpy as jnp
from jax import lax
from pdu.thermo.implicit_eos import get_thermo_properties, get_internal_energy_pt

"""
实现爆轰波前缘的 Rankine-Hugoniot 跳跃条件计算 (Von Neumann Spike)
"""

def compute_vn_spike(D, rho0, e0, P0, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params):
    """
    求解激波跳跃方程组，寻找冻结状态下的 (rho_1, u_1, T_1)
    """
    # V11 Phase 4: Standardize to SI (kg, m, s, Pa, J/kg)
    rho0_si = rho0 * 1000.0  # g/cm3 -> kg/m3
    MassFlux = rho0_si * D
    MomFlux = P0 + rho0_si * D**2
    H0 = e0 + P0 / rho0_si # J/kg
    EnergyFlux = H0 + 0.5 * D**2

    # V11 Phase 8: Baseline-Neutral Energy Matching
    # Reactant thermal energy at shock state = Reactant thermal energy at ref + Shock work
    # We use the Product-EOS as a proxy for the Reactant-EOS shape, 
    # but we must subtract the 298K baseline to avoid absolute offset issues.
    e_ref = get_internal_energy_pt(jnp.asarray(rho0), 298.0, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
    h_ref = e_ref + P0 / rho0_si
    
    def residual(rho_test):
        rho_si = rho_test * 1000.0
        u_test = MassFlux / rho_si
        P_test = MomFlux - rho_si * u_test**2
        
        # Shock Enthalpy Delta (SI J/kg)
        dh_target = (EnergyFlux - 0.5 * u_test**2) - h_ref
        
        def h_residual(t_test):
            # Calculate enthalpy delta using the EOS
            p_c, _ = get_thermo_properties(jnp.asarray(rho_test), jnp.asarray(t_test), atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
            u_c = get_internal_energy_pt(jnp.asarray(rho_test), jnp.asarray(t_test), atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
            h_c = u_c + p_c / (rho_si + 1e-10)
            dh_calc = h_c - h_ref
            return (dh_calc - dh_target) / (jnp.abs(dh_target) + 1.0)

        def find_t(l_t, h_t):
            def t_step(j, b):
                lt, ht = b
                mt = (lt + ht) / 2.0
                return jnp.where(h_residual(mt) < 0, mt, lt), jnp.where(h_residual(mt) < 0, ht, mt)
            return lax.fori_loop(0, 18, t_step, (l_t, h_t))[0]
            
        T_vn = find_t(300.0, 6000.0)
        p_final, _ = get_thermo_properties(jnp.asarray(rho_test), T_vn, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
        return (p_final - P_test) / (P_test + 1e-10)

    # 二分法寻找 rho_1
    # 对于高强激波，rho_1/rho_0 约 1.4-1.6
    def find_rho(lo, hi):
        def step(i, bounds):
            l, h = bounds
            m = (l + h) / 2.0
            # 压力残差在冲击波分支通常随密度单调上升
            return jnp.where(residual(m) > 0, l, m), jnp.where(residual(m) > 0, m, h)
        final_bounds = lax.fori_loop(0, 18, step, (lo, hi))
        return (final_bounds[0] + final_bounds[1]) / 2.0

    rho_vn = find_rho(rho0 * 1.3, rho0 * 1.7) # 约束在高压物理分支
    rho_vn_si = rho_vn * 1000.0
    u_vn = MassFlux / rho_vn_si
    
    # 最终获取 T_vn
    P_vn_target = MomFlux - rho_vn_si * u_vn**2
    h_target_final = EnergyFlux - 0.5 * u_vn**2
    
    def final_t_search(rho_val, h_val):
        dh_target = h_val - h_ref
        def res_t(t):
            u_c = get_internal_energy_pt(jnp.asarray(rho_val), jnp.asarray(t), atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
            p_c, _ = get_thermo_properties(jnp.asarray(rho_val), jnp.asarray(t), atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
            h_c = u_c + p_c / (rho_val * 1000.0 + 1e-10)
            return (h_c - h_ref) - dh_target
        l, h = 300.0, 6000.0
        for _ in range(18):
            m = (l + h) / 2.0
            l, h = jnp.where(res_t(m) < 0, m, l), jnp.where(res_t(m) < 0, h, m)
        return (l + h) / 2.0
    
    T_vn = final_t_search(rho_vn, h_target_final)
    
    return rho_vn, u_vn, T_vn