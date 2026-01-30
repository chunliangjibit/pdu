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
    jax.debug.print("JumpConditions: rho0_si={r}, D={d}, MassFlux={m}", r=rho0_si, d=D, m=MassFlux)
    MomFlux = P0 + rho0_si * D**2
    H0 = e0 + P0 / rho0_si # J/kg
    EnergyFlux = H0 + 0.5 * D**2
    
    def residual(rho_test):
        # rho_test is in g/cm3
        rho_si = rho_test * 1000.0
        u_test = MassFlux / rho_si
        P_test = MomFlux - rho_si * u_test**2
        h_target = EnergyFlux - 0.5 * u_test**2
        
        # 寻找 T 使得 H(rho, T) = h_target
        def h_residual(t_test):
            p_c, _ = get_thermo_properties(jnp.asarray(rho_test), jnp.asarray(t_test), atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
            u_c = get_internal_energy_pt(jnp.asarray(rho_test), jnp.asarray(t_test), atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
            h_calc = u_c + p_c / (rho_si + 1e-10) 
            return (h_calc - h_target) / h_target

        def find_t(l_t, h_t):
            def t_step(j, b):
                lt, ht = b
                mt = (lt + ht) / 2.0
                return jnp.where(h_residual(mt) < 0, mt, lt), jnp.where(h_residual(mt) < 0, ht, mt)
            return lax.fori_loop(0, 18, t_step, (l_t, h_t))[0]
            
        T_vn = find_t(2000.0, 4500.0)
        p_final, _ = get_thermo_properties(jnp.asarray(rho_test), T_vn, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
        # 残差：EOS 压力 vs 动量平衡压力
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
        def res_t(t):
            u_c = get_internal_energy_pt(jnp.asarray(rho_val), jnp.asarray(t), atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
            p_c, _ = get_thermo_properties(jnp.asarray(rho_val), jnp.asarray(t), atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
            return u_c + p_c / (rho_val * 1000.0 + 1e-10) - h_val
        l, h = 2000.0, 4500.0
        for _ in range(18):
            m = (l + h) / 2.0
            l, h = jnp.where(res_t(m) < 0, m, l), jnp.where(res_t(m) < 0, h, m)
        return (l + h) / 2.0
    
    T_vn = final_t_search(rho_vn, h_target_final)
    
    return rho_vn, u_vn, T_vn