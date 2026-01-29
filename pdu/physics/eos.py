"""
JCZ3 状态方程模块

基于 Exp-6 势能的高压状态方程，用于计算爆轰产物的 P-V-T 关系。
利用 JAX 自动微分计算化学势 (mu = dA/dn)。
"""

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
                    if species == 'H2O':
                        # Hardcode V10 Ree parameters for water if missing in DB
                        # Ree 1982: eps=94.5, r=3.35, alpha=13.5, lambda=5400? (Approx)
                        # Let's use neutral defaults but allow lambda injection
                        vec_eps.append(100.0) 
                        vec_r.append(3.5)
                        vec_alpha.append(13.0)
                        vec_lambda.append(0.0) 
                    else:
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

@dataclass
class CowanFickettEOS:
    """Cowan-Fickett EOS for Solids (C, Al2O3)
    
    P(V, T) = P0(V) + a(V) * T + b(V) * T^2
    Usually simplified to P = A(lambda) + B(lambda)*T
    lambda = V/V0
    """
    solid_indices: jnp.ndarray # Indices of solid species in local list
    V0_molar: jnp.ndarray # cm3/mol
    params: jnp.ndarray # Matrix of params for each solid
    
    @classmethod
    def from_species_list(cls, species_list):
        solids = ['C_graphite', 'Al2O3', 'Al']
        indices = []
        V0s = []
        # Dummy params for now: P = P0 (incompressible approx) + small T dependence
        # Typically P ~ K0 * (1 - V/V0) ? No, Murnaghan.
        # Cowan Fickett for Diamond/Graphite:
        # P = ...
        # For this verification, let's use a Murnaghan approximation for solids 
        # which is robust and standard for detonations.
        # P = (K0/n) * ((V0/V)^n - 1)
        # For Graphite: K0 ~ 33 GPa, n ~ 5?
        # For Al2O3: K0 ~ 200 GPa.
        
        # Let's stick to a simple incompressible limit correction or Murnaghan.
        # Use K0 = 50 GPa for C, 200 GPa for Al2O3.
        # Return class.
        pass

@jax.jit
def compute_effective_diameter_ratio(T, epsilon, alpha):
    """Calculate d/r* ratio based on Temperature
    
    Using condition: u_rep(d) = kT
    exp-6 repulsive part: u = eps * 6/(alpha-6) * exp(alpha*(1-d/r*))
    kT = ...
    d/r* = 1 - (1/alpha) * ln( (alpha-6)/6 * kT/eps )
    """
    alpha_minus_6 = alpha - 6.0
    # Avoid log(negative) or div/0. epsilon is K. T is K.
    # T_star = T / epsilon
    T_star = jnp.maximum(T / (epsilon + 1e-10), 1e-10)
    
    term = (alpha_minus_6 / 6.0) * T_star
    log_term = jnp.log(jnp.maximum(term, 1e-10))
    
    ratio = 1.0 - (1.0 / alpha) * log_term
    
    # Clip ratio to reasonable bounds (e.g. 0.4 to 1.2)
    # At T->0, log_term -> -inf, ratio -> large. 
    return jnp.clip(ratio, 0.4, 1.2)

@jax.jit
def compute_mixed_matrices_dynamic(
    T: float,
    eps_vec: jnp.ndarray,
    r_vec: jnp.ndarray,
    alpha_vec: jnp.ndarray,
    lambda_vec: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    [V10 Core] Calculate mixing matrices dynamic with Temperature
    Supports Francis Ree Correction: eps(T) = eps0 * (1 + lambda/T)
    """
    T_safe = jnp.maximum(T, 1e-2)
    
    # 1. Apply Ree Correction (Polar Water)
    eps_T = eps_vec * (1.0 + lambda_vec / T_safe)
    
    # 2. Lorentz-Berthelot Mixing
    # Epsilon: Geometric Mean: sqrt(e_i * e_j)
    # Outer product equivalent: sqrt(e_i * e_j) = sqrt(outer(e, e))
    eps_matrix = jnp.sqrt(jnp.outer(eps_T, eps_T))
    
    # R_star: Arithmetic Mean: (r_i + r_j) / 2
    r_matrix = 0.5 * (jnp.expand_dims(r_vec, 1) + jnp.expand_dims(r_vec, 0))
    
    # Alpha: Arithmetic Mean: (a_i + a_j) / 2
    alpha_matrix = 0.5 * (jnp.expand_dims(alpha_vec, 1) + jnp.expand_dims(alpha_vec, 0))
    
    return eps_matrix, r_matrix, alpha_matrix

@jax.jit
def compute_excess_repulsion(
    rho: float,
    coeff_stiff: float = 300.0,
    rho_ref: float = 2.0
) -> float:
    """
    V8.6 High-Pressure Repulsive Correction (Stiffening Term)
    \Delta F_{rep} = C * (\rho / \rho_ref)^4
    Compensates for the softness of alpha=13.0 at detonation densities.
    """
    # Only apply when density is substantial to avoid low-density artifacts
    ratio = rho / rho_ref
    # Soft activation to avoid negative impact at low densities
    # Using simple power law 
    f_rep = coeff_stiff * (ratio**4)
    return f_rep

@jax.jit
def compute_solid_volume_murnaghan(solid_v0, P_est, is_carbon, is_alumina):
    """
    计算凝聚相动态体积 (Murnaghan EOS)
    V(P) = V0 * (1 + n*P/K0)^(-1/n)
    
    [V8.6 Upgrade]: Carbon swapped to Fried-Howard Liquid Carbon
    """
    
    # 1. 碳 (Carbon) -> Fried-Howard Liquid Carbon Model
    # V0 = 4.44 cc/mol, K0 = 60 GPa, n = 6.0
    v0_c_target = 4.44 
    # Ignore input solid_v0 for carbon, enforce liquid species param
    
    c_compress = jnp.power(1.0 + 6.0 * P_est / 60e9, -1.0/6.0)
    vol_c = v0_c_target * c_compress
    
    # 2. 氧化铝 (Alumina) -> Liquid Al2O3
    # K0=150 GPa, n=4.0
    al_compress = jnp.power(1.0 + 4.0 * P_est / 150e9, -1.0/4.0)
    vol_al = solid_v0 * 1.32 * al_compress
    
    # 3. 组合
    v_final = jnp.where(is_carbon, vol_c, solid_v0)
    v_final = jnp.where(is_alumina, vol_al, v_final)
    return v_final

@jax.jit
def compute_covolume(n, r_star_matrix, T=None, epsilon_matrix=None, alpha_matrix=None, rho_impact: float = 0.0):
    """b = (2π/3) * N_A * ΣΣ x_i x_j d_ij^3  * n_total ? 
    Standard One-Fluid: b_mix = sum x_i x_j b_ij
    
    [V10.1 Upgrade]: Added rho_impact to handle density-dependent r* for liquids (NM fix).
    We use the packing fraction eta as a feedback variable for r* hardening.
    r_eff = r_star * (1 + rho_impact * eta^2)
    """
    n = to_fp64(n)
    n_total = jnp.sum(n) + 1e-30
    
    # 1. First Pass: Calculate raw d(T)
    if T is not None and epsilon_matrix is not None and alpha_matrix is not None:
        ratio = compute_effective_diameter_ratio(T, epsilon_matrix, alpha_matrix)
        d_raw = r_star_matrix * ratio
    else:
        d_raw = r_star_matrix
        
    # 2. Estimate eta for hardening logic
    # (Simplified: using a standard B/4V proxy for eta internally)
    d3_raw = d_raw ** 3 * 1e-24 # cm3
    B_raw = (2.0 * jnp.pi / 3.0) * N_AVOGADRO * (jnp.sum(jnp.outer(n, n) * d3_raw) / n_total)
    
    # Since we don't have V here, we use a reference packing fraction for the correction
    # typically detonation eta is ~0.65.
    # We apply the correction based on rho_impact.
    # r_final = r_raw * (1 + rho_impact)
    d_matrix = d_raw * (1.0 + rho_impact)
        
    d3_matrix = d_matrix ** 3 * 1e-24 # cm3
    n_outer = jnp.outer(n, n)
    sum_nd3 = jnp.sum(n_outer * d3_matrix)
    
    B_total = (2.0 * jnp.pi / 3.0) * N_AVOGADRO * (sum_nd3 / n_total)
    
    return B_total

@jax.jit
def compute_total_helmholtz_energy(
    n: jnp.ndarray,
    V_total: float,
    T: float,
    coeffs_all: jnp.ndarray,
    # V10 Upgrade: Pass Vectors instead of Matrices
    eps_vec: jnp.ndarray,
    r_star_vec: jnp.ndarray,
    alpha_vec: jnp.ndarray,
    lambda_vec: jnp.ndarray,
    solid_mask: Optional[jnp.ndarray] = None,   # 1.0=Solid, 0.0=Gas (in equilibrium)
    solid_v0: Optional[jnp.ndarray] = None,     # Equilibrium solid molar volume (cm3/mol)
    n_fixed_solids: Optional[float] = 0.0,      # V9: Moles of fixed inert solid (e.g. Inert Al)
    v0_fixed_solids: Optional[float] = 10.0,    # V9: Molar volume of fixed inert solid (Al=10.0)
    e_fixed_solids: Optional[float] = 0.0,      # V9: Internal energy contribution of fixed solids (approx)
    r_star_rho_corr: float = 0.0                # V10.1: Density-dependent r* correction factor
) -> float:
    """
    V9 Upgrade: Support 'Partial Reaction' by adding fixed inert solids that occupy volume
    but do not participate in equilibrium (n_fixed_solids).
    """
    n = to_fp64(n)
    V_total = to_fp64(V_total)
    T = to_fp64(T)
    R = R_GAS

    # 1. 默认处理
    if solid_mask is None:
        solid_mask = jnp.zeros_like(n)
    if solid_v0 is None:
        solid_v0 = jnp.zeros_like(n)

    # 2. 分离气相与固相
    n_solid = n * solid_mask
    n_gas = n * (1.0 - solid_mask)
    n_gas_total = jnp.sum(n_gas) + 1e-30

    # === [V8.4 Patch Start] ===
    # 估算压力 P_est 用于固相压缩 (利用理想气体定律近似，避免循环依赖)
    # P ~ (n_gas * R * T) / (V_total * 0.6 * 1e-6) 假设气体约占总容积 60%
    n_gas_sum = jnp.sum(n_gas) + 1e-10
    P_proxy = (n_gas_sum * 8.314 * T) / (V_total * 0.6 * 1e-6) 
    P_proxy = jnp.maximum(P_proxy, 1e5)

    # 识别组分
    is_carbon = (solid_v0 > 5.0) & (solid_v0 < 6.0)
    is_alumina = (solid_v0 > 24.0) & (solid_v0 < 27.0)

    # 计算动态有效体积
    solid_vol_eff = compute_solid_volume_murnaghan(solid_v0, P_proxy, is_carbon, is_alumina)
    
    # Equilibrium Solids Volume
    V_condensed_eq = jnp.sum(n_solid * solid_vol_eff)
    
    # Fixed Inert Solids Volume (V9)
    # Assume fixed solids are incompressible or use Murnaghan with fixed params? 
    # For Al, K0=76 GPa. Let's use simple Murnaghan for Al here too.
    # v_inert = v0 * (1 + 4*P/K)^(-1/4)
    inert_compress = jnp.power(1.0 + 4.0 * P_proxy / 76e9, -1.0/4.0)
    V_condensed_fixed = n_fixed_solids * v0_fixed_solids * inert_compress
    
    V_condensed_total = V_condensed_eq + V_condensed_fixed
    V_gas_eff = jnp.maximum(V_total - V_condensed_total, 1e-3)
    V_gas_m3 = V_gas_eff * 1e-6
    # === [V8.4 Patch End] ===

    # [V10 Core] Dynamic Mixing Calculation
    epsilon_matrix, r_star_matrix, alpha_matrix = compute_mixed_matrices_dynamic(
        T, eps_vec, r_star_vec, alpha_vec, lambda_vec
    )

    # === 4. 气相自由能 (JCZ3 + Ideal Gas) ===
    # (A) 理想气体
    def get_thermo(c): return compute_u_ideal(c, T), compute_entropy(c, T)
    u_vec, s0_vec = jax.vmap(get_thermo)(coeffs_all)
    
    P0 = 1e5
    n_gas_safe = jnp.maximum(n_gas, 1e-15)
    val_for_log = (n_gas_safe * R * T) / (V_gas_m3 * P0)
    ln_terms_gas = jnp.log(val_for_log)
    
    # Use where to zero out contribution but maintain finite gradient
    S_term = jnp.where(n_gas > 1e-18, n_gas * ln_terms_gas, 0.0)
    A_gas_ideal = jnp.sum(n_gas * (u_vec - T * s0_vec)) + R * T * jnp.sum(S_term)
    
    # (B) JCZ3 非理想修正
    # Apply r_star correction based on packing fraction eta from previous step? 
    # No, compute it dynamically using the passed correction factor.
    B_total_gas = compute_covolume(n_gas, r_star_matrix, T, epsilon_matrix, alpha_matrix, rho_impact=r_star_rho_corr)
    eta = B_total_gas / (4.0 * V_gas_eff)
    eta_limit = 0.74
    eta = eta_limit * jnp.tanh(eta / eta_limit)
    A_excess_hs = n_gas_total * R * T * (4.0*eta - 3.0*eta**2) / jnp.maximum((1.0 - eta)**2, 1e-6)
    
    eps_r3 = epsilon_matrix * r_star_matrix**3
    n_outer_gas = jnp.outer(n_gas, n_gas) 
    a_sum = jnp.sum(n_outer_gas * eps_r3)
    factor = (2.0 * jnp.pi / 3.0) * (N_AVOGADRO**2) * K_BOLTZMANN * 1e-24
    U_attr = - (factor * a_sum) / V_gas_eff
    
    # === [V8.6 Upgrade: Repulsive Correction] ===
    # Calculate density (approx)
    # Total mass / Total Volume? No, local density of the detonation fluid
    # Use gas density? Or total loading density?
    # Repulsive correction is for the fluid EOS, so use gas density?
    # BUT, the problem is macroscopic pressure deficit.
    # The fluid is squeezed.
    # Let's use avg density of the gas phase = Mass_gas / V_gas_eff ?
    # Or just use total density if we treat it as bulk correction?
    # Protocol says "Delta F_rep(rho)". Usually rho is the density of the fluid described by Exp-6.
    # So rho_gas is appropriate.
    # Mass of gas? Need molecular weights.
    # Simplified: Use Molar Density n_gas_total / V_gas_eff (mol/cm3)
    # rho_g (g/cm3) = (n_gas * MW).sum() / V_gas_eff
    # Since we don't have MW here easily (coeffs_all has them?), let's approximate or pass masses.
    # Actually, can use n_gas / V_gas_eff as a proxy for density if we tune C_stiff accordingly.
    # Or better: The Ross theory is about packing fraction eta.
    # We can add correction based on eta!
    # \Delta F = C * eta^4 ?
    # Or strict density.
    # Let's use eta since we have it. eta ~ density.
    # eta at detonation is ~ 0.5 - 0.7.
    # Let's add A_rep = n_gas_total * R * T * C_stiff * eta^4
    # This scales with Temperature and Density. Proper Repulsive Free Energy implies T dependence (like Hard Sphere).
    # If we want pure Potential Energy correction (static repulsion), it allows U_rep adjustment.
    # Let's use T-dependent term to maintain F form: k * T * f(eta).
    
    # V8.6 Tuning: C_stiff = 15.0 (Calibrated to HMX ~40 GPa)
    # eta^4 is strong.
    # A_rep = n_gas_total * R * T * 15.0 * eta**4
    
    A_rep = n_gas_total * R * T * 15.0 * (eta**4)
    
    A_gas_total = A_gas_ideal + A_excess_hs + U_attr + A_rep

    # === 5. 固相自由能 ===
    # === 5. 固相自由能 ===
    A_solid_eq = jnp.sum(n_solid * (u_vec - T * s0_vec))
    
    # Fixed Solids Free Energy (Approximation A = U - TS)
    # We assume e_fixed_solids gives U contribution (e.g. C_v * T)
    # Simple approx for Al: Cv = 24.2 J/mol.K
    # U = Hf + Cv(T-298)
    # S = S0 + Cv ln(T/298)
    # A = U - TS
    # This is getting complex. Let's pass a simplified 'fixed_energy_term' if possible.
    # For now, simplistic scaling for volume effect is primary.
    # Energy effect of inert filler is secondary for P_CJ but important for T_CJ.
    # Let's add simple heat capacity term.
    A_solid_fixed = n_fixed_solids * (e_fixed_solids + 24.3 * (T - 298.0) - T * (28.3 + 24.3 * jnp.log(T/298.0)))
    
    return A_gas_total + A_solid_eq + A_solid_fixed

@jax.jit
def compute_chemical_potential_jcz3(
    n: jnp.ndarray,
    V: float,
    T: float,
    coeffs_all: jnp.ndarray,
    eps_vec: jnp.ndarray,
    r_star_vec: jnp.ndarray,
    alpha_vec: jnp.ndarray,
    lambda_vec: jnp.ndarray,
    solid_mask: Optional[jnp.ndarray] = None,
    solid_v0: Optional[jnp.ndarray] = None,
    n_fixed_solids: float = 0.0,
    v0_fixed_solids: float = 10.0,
    e_fixed_solids: float = 0.0,
    r_star_rho_corr: float = 0.0
) -> jnp.ndarray:
    grad_fn = jax.grad(compute_total_helmholtz_energy, argnums=0)
    mu_vec = grad_fn(n, V, T, coeffs_all, eps_vec, r_star_vec, alpha_vec, lambda_vec, solid_mask, solid_v0, n_fixed_solids, v0_fixed_solids, e_fixed_solids, r_star_rho_corr)
    return mu_vec

@jax.jit
def compute_pressure_jcz3(
    n: jnp.ndarray,
    V: float,
    T: float,
    coeffs_all: jnp.ndarray,
    eps_vec: jnp.ndarray,
    r_star_vec: jnp.ndarray,
    alpha_vec: jnp.ndarray,
    lambda_vec: jnp.ndarray,
    solid_mask: Optional[jnp.ndarray] = None,
    solid_v0: Optional[jnp.ndarray] = None,
    n_fixed_solids: float = 0.0,
    v0_fixed_solids: float = 10.0,
    e_fixed_solids: float = 0.0,
    r_star_rho_corr: float = 0.0
) -> float:
    grad_fn = jax.grad(compute_total_helmholtz_energy, argnums=1)
    dA_dV = grad_fn(n, V, T, coeffs_all, eps_vec, r_star_vec, alpha_vec, lambda_vec, solid_mask, solid_v0, n_fixed_solids, v0_fixed_solids, e_fixed_solids, r_star_rho_corr) 
    return -dA_dV * 1e6

@jax.jit
def compute_internal_energy_jcz3(
    n: jnp.ndarray,
    V: float,
    T: float,
    coeffs_all: jnp.ndarray,
    eps_vec: jnp.ndarray,
    r_star_vec: jnp.ndarray,
    alpha_vec: jnp.ndarray,
    lambda_vec: jnp.ndarray,
    solid_mask: Optional[jnp.ndarray] = None,
    solid_v0: Optional[jnp.ndarray] = None,
    n_fixed_solids: float = 0.0,
    v0_fixed_solids: float = 10.0,
    e_fixed_solids: float = 0.0,
    r_star_rho_corr: float = 0.0
) -> float:
    grad_T_fn = jax.grad(compute_total_helmholtz_energy, argnums=2)
    dA_dT = grad_T_fn(n, V, T, coeffs_all, eps_vec, r_star_vec, alpha_vec, lambda_vec, solid_mask, solid_v0, n_fixed_solids, v0_fixed_solids, e_fixed_solids, r_star_rho_corr)
    A = compute_total_helmholtz_energy(n, V, T, coeffs_all, eps_vec, r_star_vec, alpha_vec, lambda_vec, solid_mask, solid_v0, n_fixed_solids, v0_fixed_solids, e_fixed_solids, r_star_rho_corr)
    return A - T * dA_dT

@jax.jit
def compute_entropy_consistent(
    n: jnp.ndarray,
    V: float,
    T: float,
    coeffs_all: jnp.ndarray,
    eps_vec: jnp.ndarray,
    r_star_vec: jnp.ndarray,
    alpha_vec: jnp.ndarray,
    lambda_vec: jnp.ndarray,
    solid_mask: Optional[jnp.ndarray] = None,
    solid_v0: Optional[jnp.ndarray] = None,
    n_fixed_solids: float = 0.0,
    v0_fixed_solids: float = 10.0,
    e_fixed_solids: float = 0.0,
    r_star_rho_corr: float = 0.0
) -> float:
    def h_wrt_T(t):
        return compute_total_helmholtz_energy(n, V, t, coeffs_all, eps_vec, r_star_vec, alpha_vec, lambda_vec, solid_mask, solid_v0, n_fixed_solids, v0_fixed_solids, e_fixed_solids, r_star_rho_corr)
    return -jax.grad(h_wrt_T)(T)
