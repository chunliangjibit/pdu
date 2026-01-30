from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from functools import partial

from pdu.utils.precision import to_fp32, to_fp64, R_GAS
from pdu.physics.thermo import compute_entropy, compute_internal_energy as compute_u_ideal

# 玻尔兹曼常数 (J/K)
K_BOLTZMANN = 1.380649e-23
# 阿伏伽德罗常数
N_AVOGADRO = 6.02214076e23

@jax.jit
def smooth_floor(x, xmin, w):
    # C1 光滑下界：>= xmin
    return xmin + jax.nn.softplus((x - xmin) / w) * w

@jax.jit
def smooth_cap(x, xmax, w):
    # C1 光滑上界：<= xmax
    return xmax - jax.nn.softplus((xmax - x) / w) * w

@jax.jit
def smooth_switch(T, T0=1000.0, width=30.0):
    # Sigmoid 平滑切换 (0 -> 1)
    return jax.nn.sigmoid((T - T0) / width)

@dataclass
class JCZ3EOS:
    """JCZ3 状态方程类 (Refactored for V10 Dynamic Mixing)"""
    species_names: tuple
    # Pure component parameters (Vector form)
    eps_vec: jnp.ndarray      # epsilon/k (K)
    r_star_vec: jnp.ndarray   # r* (Angstrom)
    alpha_vec: jnp.ndarray    # alpha (Repulsive steepness)
    lambda_vec: jnp.ndarray   # [V10] Francis Ree Polar correction factor (K)
    coeffs_all: jnp.ndarray 

    @classmethod
    def from_species_list(cls, species_list: list, coeffs_all: jnp.ndarray) -> "JCZ3EOS":
        """从物种列表构建 EOS (V10 Version)"""
        import json
        import os
        
        params = {}
        # Try load from json
        json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'jcz3_params.json')
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                db = data.get('species', {})
            
            vec_eps = []
            vec_r = []
            vec_alpha = []
            vec_lambda = []
            
            for species in species_list:
                if species in db:
                    p = db[species]
                    vec_eps.append(p.get('epsilon_over_k', 100.0))
                    vec_r.append(p.get('r_star', 3.5))
                    vec_alpha.append(p.get('alpha', 13.0))
                    # Check for 'lambda' polar parameter, default 0.0
                    vec_lambda.append(p.get('lambda_ree', 0.0)) 
                else:
                    # Default / Fallback
                    vec_eps.append(100.0)
                    vec_r.append(3.5)
                    vec_alpha.append(13.0)
                    vec_lambda.append(0.0)

        except Exception as e:
             # Fallback if file load fails
             print(f"Warning: Failed to load JCZ3 DB ({e}), using defaults.")
             vec_eps = [100.0] * len(species_list)
             vec_r = [3.5] * len(species_list)
             vec_alpha = [13.0] * len(species_list)
             vec_lambda = [0.0] * len(species_list)

        return cls(
            species_names=tuple(species_list),
            eps_vec=jnp.array(vec_eps),
            r_star_vec=jnp.array(vec_r),
            alpha_vec=jnp.array(vec_alpha),
            lambda_vec=jnp.array(vec_lambda),
            coeffs_all=coeffs_all
        )

# Hoisted vmap for better compilation stability
def _get_thermo_single(coeffs_low, coeffs_high, T):
    u_low = compute_u_ideal(coeffs_low, T)
    s_low = compute_entropy(coeffs_low, T)
    u_high = compute_u_ideal(coeffs_high, T)
    s_high = compute_entropy(coeffs_high, T)
    
    # Patch B: Sigmoid 平滑过渡，消除 T=1000K 处的导数尖峰
    w = smooth_switch(T, 1000.0, 30.0)
    u_val = (1.0 - w) * u_low + w * u_high
    s_val = (1.0 - w) * s_low + w * s_high
    return u_val, s_val

_get_thermo_vec = jax.vmap(_get_thermo_single, in_axes=(0, 0, None))

@jax.jit
def compute_effective_diameter_ratio(T, epsilon, alpha):
    """Calculate d/r* ratio based on Temperature"""
    alpha_minus_6 = alpha - 6.0
    T_star = jnp.maximum(T / (epsilon + 1e-10), 1e-10)
    term = (alpha_minus_6 / 6.0) * T_star
    log_term = jnp.log(jnp.maximum(term, 1e-10))
    ratio = 1.0 - (1.0 / alpha) * log_term
    
    # Patch A: Smooth Clip for diameter ratio
    ratio = smooth_floor(ratio, 0.4, 0.02)
    ratio = smooth_cap(ratio, 1.2, 0.02)
    return ratio

@jax.jit
def compute_polar_epsilon_ree_ross(T, eps0, lambda_ree):
    """Ree Polar Correction (V10.0 simplified)"""
    T_safe = jnp.maximum(T, 100.0)
    return eps0 * (1.0 + lambda_ree / T_safe)

@jax.jit
def compute_mixed_matrices_dynamic(T, eps_vec, r_vec, alpha_vec, lambda_vec):
    """Calculate mixing matrices"""
    T_safe = jnp.maximum(T, 1e-2)
    eps_T = eps_vec * (1.0 + lambda_vec / T_safe)
    eps_matrix = jnp.sqrt(jnp.outer(eps_T, eps_T))
    r_matrix = 0.5 * (jnp.expand_dims(r_vec, 1) + jnp.expand_dims(r_vec, 0))
    alpha_matrix = 0.5 * (jnp.expand_dims(alpha_vec, 1) + jnp.expand_dims(alpha_vec, 0))
    return eps_matrix, r_matrix, alpha_matrix

@jax.jit
def compute_solid_volume_murnaghan(solid_v0, P_est, is_carbon, is_alumina):
    """凝聚相 Murnaghan EOS"""
    v0_c_target = 4.44 
    c_compress = jnp.power(1.0 + 6.0 * P_est / 60e9, -1.0/6.0)
    vol_c = v0_c_target * c_compress
    al_compress = jnp.power(1.0 + 4.0 * P_est / 150e9, -1.0/4.0)
    vol_al = solid_v0 * 1.32 * al_compress
    v_final = jnp.where(is_carbon, vol_c, solid_v0)
    v_final = jnp.where(is_alumina, vol_al, v_final)
    return v_final

@jax.jit
def compute_covolume(n, r_star_matrix, T, epsilon_matrix, alpha_matrix, rho_impact=0.0):
    """计算混合共体积 B"""
    n = to_fp64(n)
    n_total = jnp.sum(n) + 1e-30
    ratio = compute_effective_diameter_ratio(T, epsilon_matrix, alpha_matrix)
    d_raw = r_star_matrix * ratio
    d_matrix = d_raw * (1.0 + rho_impact)
    d3_matrix = d_matrix ** 3 * 1e-24 # cm3
    sum_nd3 = jnp.sum(jnp.outer(n, n) * d3_matrix)
    return (2.0 * jnp.pi / 3.0) * N_AVOGADRO * (sum_nd3 / n_total)

@jax.jit
def alpha_hardening(rho, alpha0, k=0.0, rho_crit=2.2, w=0.1):
    """
    密度依赖的 alpha 硬化 (V11.0 Disable for baseline)
    """
    return alpha0 + k * jax.nn.softplus((rho - rho_crit) / w)

def compute_total_helmholtz_energy(
    n, V_total, T, coeffs_low, coeffs_high,
    eps_vec, r_star_vec, alpha_vec, lambda_vec,
    solid_mask, solid_v0,
    n_fixed_solids=0.0, v0_fixed_solids=10.0, e_fixed_solids=0.0,
    r_star_rho_corr=0.0,
    mw_avg=1.0
):
    # V11 Phase 4: Enforce FP64 for thermodynamic consistency at super-high density
    n = jnp.asarray(n, dtype=jnp.float64)
    V_total = jnp.asarray(V_total, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)
    R = R_GAS
    
    rho_macro = mw_avg / (V_total + 1e-10)
    alpha_hardened = alpha_hardening(rho_macro, alpha_vec)
    
    n_solid, n_gas = n * solid_mask, n * (1.0 - solid_mask)
    n_gas_total = jnp.sum(n_gas) + 1e-30
    n_gas_total = jnp.sum(n_gas) + 1e-30
    # jax.debug.print("rho={r}, n_gas_total={ngt}", r=rho_macro, ngt=n_gas_total)
    
    P_proxy = (n_gas_total * 8.314 * T) / (V_total * 0.6 * 1e-6)
    
    P_proxy = (n_gas_total * 8.314 * T) / (V_total * 0.6 * 1e-6) 
    P_proxy = jnp.maximum(P_proxy, 1e5)
    is_carbon = (solid_v0 > 5.0) & (solid_v0 < 6.0)
    is_alumina = (solid_v0 > 24.0) & (solid_v0 < 27.0)
    solid_vol_eff = compute_solid_volume_murnaghan(solid_v0, P_proxy, is_carbon, is_alumina)
    
    V_condensed_eq = jnp.sum(n_solid * solid_vol_eff)
    inert_compress = jnp.power(1.0 + 4.0 * P_proxy / 76e9, -1.0/4.0)
    V_condensed_fixed = n_fixed_solids * v0_fixed_solids * inert_compress
    V_condensed_total = V_condensed_eq + V_condensed_fixed
    
    # Patch A: V_gas_eff 光滑地板 (V_MIN=1e-6 cm3)
    V_gas_raw = V_total - V_condensed_total
    V_gas_eff = smooth_floor(V_gas_raw, 1e-6, 1e-6)
    V_gas_m3 = V_gas_eff * 1e-6

    # 气相非理想
    eps_mat, r_mat, alpha_mat = compute_mixed_matrices_dynamic(T, eps_vec, r_star_vec, alpha_hardened, lambda_vec)
    u_vec, s0_vec = _get_thermo_vec(coeffs_low, coeffs_high, T)
    
    n_gas_safe = jnp.maximum(n_gas, 1e-15)
    A_gas_ideal = jnp.sum(n_gas * (u_vec - T * s0_vec)) + R * T * jnp.sum(jnp.where(n_gas > 1e-18, n_gas * jnp.log((n_gas_safe * R * T) / (V_gas_m3 * 1e5)), 0.0))
    
    B_total_gas = compute_covolume(n_gas, r_mat, T, eps_mat, alpha_mat, r_star_rho_corr)
    eta_raw = B_total_gas / (4.0 * V_gas_eff)
    
    # Patch A: Safe CS + Soft Barrier for eta > 0.95
    ETA_CAP = 0.95
    ETA_CAP_W = 0.005
    
    # CS 部分只在 safe 域内有效 (<0.95)
    eta_cs = smooth_cap(eta_raw, ETA_CAP, ETA_CAP_W)
    one_minus = smooth_floor(1.0 - eta_cs, 1e-6, 1e-6)
    f_cs = (4.0 * eta_cs - 3.0 * eta_cs * eta_cs) / (one_minus * one_minus)
    A_cs = n_gas_total * R * T * f_cs
    
    # Barrier 部分: 对 overshoot 进行惩罚
    OVER_W = 0.02
    over = jax.nn.softplus((eta_raw - ETA_CAP) / OVER_W) * OVER_W
    K_ETA = 100.0
    A_bar = n_gas_total * R * T * K_ETA * (over / OVER_W) ** 2
    
    A_excess_hs = A_cs + A_bar
    A_excess_hs = A_cs + A_bar
    # jax.debug.print("T={t}, V_gas={v}, eta_raw={e}, A_hs={a}", t=T, v=V_gas_eff, e=eta_raw, a=A_excess_hs)
    
    U_attr = - ((2.0 * jnp.pi / 3.0) * (N_AVOGADRO**2) * K_BOLTZMANN * 1e-24 * jnp.sum(jnp.outer(n_gas, n_gas) * eps_mat * r_mat**3)) / V_gas_eff
    
    U_attr = - ((2.0 * jnp.pi / 3.0) * (N_AVOGADRO**2) * K_BOLTZMANN * 1e-24 * jnp.sum(jnp.outer(n_gas, n_gas) * eps_mat * r_mat**3)) / V_gas_eff
    
    A_rep = 0.0 # Deprecated, merged into A_bar
    A_gas_total = A_gas_ideal + A_excess_hs + U_attr

    A_solid_eq = jnp.sum(n_solid * (u_vec - T * s0_vec))
    cv_al_eff = 45.0
    A_solid_fixed = n_fixed_solids * (e_fixed_solids + cv_al_eff * (T - 298.0) - T * (28.3 + cv_al_eff * jnp.log(jnp.maximum(T / 298.0, 1e-3))))
    
    A_val = A_gas_total + A_solid_eq + A_solid_fixed
    A_val = A_gas_total + A_solid_eq + A_solid_fixed
    # jax.debug.print("A_final={a}, A_rep={rep}", a=A_val, rep=A_rep)
    return A_val
    return A_val

# Hoisted Differentiation: Define core gradient functions once at top level
_energy_grad_n_raw = jax.grad(compute_total_helmholtz_energy, argnums=0)
_energy_grad_V_raw = jax.grad(compute_total_helmholtz_energy, argnums=1)
_energy_grad_T_raw = jax.grad(compute_total_helmholtz_energy, argnums=2)
_energy_val_grad_n_raw = jax.value_and_grad(compute_total_helmholtz_energy, argnums=0)

@jax.jit
def compute_pressure_jcz3(n, V_total, T, coeffs_low, coeffs_high, *args):
    """Jit-friendly pressure calculation using hoisted grad"""
    return -_energy_grad_V_raw(n, V_total, T, coeffs_low, coeffs_high, *args) * 1e6

@jax.jit
def compute_internal_energy_jcz3(n, V_total, T, coeffs_low, coeffs_high, *args):
    """Jit-friendly internal energy calculation"""
    A = compute_total_helmholtz_energy(n, V_total, T, coeffs_low, coeffs_high, *args)
    grad_T = _energy_grad_T_raw(n, V_total, T, coeffs_low, coeffs_high, *args)
    U_raw = A - T * grad_T
    
    # Cage Effect (Physics correction)
    P_pa = -_energy_grad_V_raw(n, V_total, T, coeffs_low, coeffs_high, *args) * 1e6
    P_gpa = P_pa * 1e-9
    corr_factor = 1.0 / (1.0 + (P_gpa / 60.0)**2 * 0.15)
    return U_raw * corr_factor

@jax.jit
def compute_chemical_potential_jcz3(n, V_total, T, coeffs_low, coeffs_high, *args):
    """Jit-friendly chemical potential using hoisted grad"""
    return _energy_grad_n_raw(n, V_total, T, coeffs_low, coeffs_high, *args)

@jax.jit
def compute_entropy_consistent(n, V_total, T, coeffs_low, coeffs_high, *args):
    """Jit-friendly entropy calculation"""
    return -_energy_grad_T_raw(n, V_total, T, coeffs_low, coeffs_high, *args)

@jax.jit
def compute_A_and_mu_jcz3(n, V_total, T, coeffs_low, coeffs_high, *args):
    """Optimized A and mu calculation using value_and_grad"""
    return _energy_val_grad_n_raw(n, V_total, T, coeffs_low, coeffs_high, *args)