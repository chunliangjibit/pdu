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
    """JCZ3 状态方程类"""
    species_names: tuple
    epsilon_matrix: jnp.ndarray
    r_star_matrix: jnp.ndarray
    alpha_matrix: jnp.ndarray
    coeffs_all: jnp.ndarray 

    @classmethod
    def from_species_list(cls, species_list: list, coeffs_all: jnp.ndarray) -> "JCZ3EOS":
        """从物种列表构建 EOS"""
        from pdu.data.products import get_exp6_params
        from pdu.physics.potential import build_mixing_matrices
        
        import json
        import os
        
        params = {}
        # Try load from json
        json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'jcz3_params.json')
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                db = data.get('species', {})
            
            for species in species_list:
                if species in db:
                    p = db[species]
                    params[species] = (p['epsilon_over_k'], p['r_star'], p['alpha'])
                else:
                    # Default
                    params[species] = (100.0, 3.5, 13.0)
        except:
             for species in species_list:
                params[species] = (100.0, 3.5, 13.0)

        eps_matrix, r_matrix, alpha_matrix = build_mixing_matrices(params)
        
        return cls(
            species_names=tuple(species_list),
            epsilon_matrix=eps_matrix,
            r_star_matrix=r_matrix,
            alpha_matrix=alpha_matrix,
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
def compute_covolume(n, r_star_matrix, T=None, epsilon_matrix=None, alpha_matrix=None):
    """b = (2π/3) * N_A * ΣΣ x_i x_j d_ij^3  * n_total ? 
    Standard One-Fluid: b_mix = sum x_i x_j b_ij
    Total B = n_total * b_mix = (1/n) * sum n_i n_j b_ij.
    """
    n = to_fp64(n)
    n_total = jnp.sum(n) + 1e-30
    
    # Effective diameter d(T)
    if T is not None and epsilon_matrix is not None and alpha_matrix is not None:
        ratio = compute_effective_diameter_ratio(T, epsilon_matrix, alpha_matrix)
        d_matrix = r_star_matrix * ratio
    else:
        d_matrix = r_star_matrix
        
    d3_matrix = d_matrix ** 3 * 1e-24 # cm3
    n_outer = jnp.outer(n, n)
    
    # Sum n_i n_j d^3
    sum_nd3 = jnp.sum(n_outer * d3_matrix)
    
    # b_mix_molar = (2pi/3) * N_A * (sum_nd3 / n^2)
    # Total B = n * b_mix_molar = (2pi/3) * N_A * (sum_nd3 / n)
    
    B_total = (2.0 * jnp.pi / 3.0) * N_AVOGADRO * (sum_nd3 / n_total)
    
    return B_total

@jax.jit
def compute_total_helmholtz_energy(
    n: jnp.ndarray,
    V_total: float,
    T: float,
    coeffs_all: jnp.ndarray,
    epsilon_matrix: jnp.ndarray,
    r_star_matrix: jnp.ndarray,
    alpha_matrix: jnp.ndarray,
    solid_mask: Optional[jnp.ndarray] = None,   # 1.0=Solid, 0.0=Gas
    solid_v0: Optional[jnp.ndarray] = None      # 固相摩尔体积 (cm3/mol)
) -> float:
    """
    V8 核心升级: 多相亥姆霍兹自由能 (气固分离 + 体积扣除)
    解决了固相产物被当成高压气体处理导致的"能量稀释"问题。
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

    # 3. 体积扣除 (V8.1 Fix: 固相高压体积修正)
    # 物理背景: 在爆轰压力下，碳以类金刚石形态存在 (V_eff ~ 0.65 * V_graphite)
    # Al2O3 为硬陶瓷，压缩较小 (V_eff ~ 0.92 * V_0)
    compress_factors = jnp.ones_like(solid_v0)
    # 自动识别策略 (基于 V0 值段的工程识别):
    # 碳 (Graphite V0 ~ 5.3): 压缩至金刚石密度
    compress_factors = jnp.where((solid_v0 > 5.0) & (solid_v0 < 6.0), 0.645, compress_factors)
    # 氧化铝 (Al2O3 V0 ~ 25.6): Murnaghan 压缩估计
    compress_factors = jnp.where((solid_v0 > 20.0) & (solid_v0 < 30.0), 0.92, compress_factors)
    
    # 计算有效固相体积
    V_solids_eff = jnp.sum(n_solid * solid_v0 * compress_factors)
    V_gas_eff = jnp.maximum(V_total - V_solids_eff, 1e-2) 
    V_gas_m3 = V_gas_eff * 1e-6

    # === 4. 气相自由能 (JCZ3 + Ideal Gas) ===
    # (A) 理想气体
    def get_thermo(c): return compute_u_ideal(c, T), compute_entropy(c, T)
    u_vec, s0_vec = jax.vmap(get_thermo)(coeffs_all)
    
    P0 = 1e5
    val_for_log = (n_gas * R * T) / (V_gas_m3 * P0)
    ln_terms_gas = jnp.log(jnp.maximum(val_for_log, 1e-30))
    S_correction_gas = -R * jnp.sum(jnp.where(n_gas > 1e-20, n_gas * ln_terms_gas, 0.0))
    A_gas_ideal = jnp.sum(n_gas * (u_vec - T * s0_vec)) - T * S_correction_gas
    
    # (B) JCZ3 非理想修正
    B_total_gas = compute_covolume(n_gas, r_star_matrix, T, epsilon_matrix, alpha_matrix)
    eta = B_total_gas / (4.0 * V_gas_eff)
    eta_limit = 0.74
    eta = eta_limit * jnp.tanh(eta / eta_limit)
    A_excess_hs = n_gas_total * R * T * (4.0*eta - 3.0*eta**2) / jnp.maximum((1.0 - eta)**2, 1e-6)
    
    eps_r3 = epsilon_matrix * r_star_matrix**3
    n_outer_gas = jnp.outer(n_gas, n_gas) 
    a_sum = jnp.sum(n_outer_gas * eps_r3)
    factor = (2.0 * jnp.pi / 3.0) * (N_AVOGADRO**2) * K_BOLTZMANN * 1e-24
    U_attr = - (factor * a_sum) / V_gas_eff
    
    A_gas_total = A_gas_ideal + A_excess_hs + U_attr

    # === 5. 固相自由能 ===
    A_solid = jnp.sum(n_solid * (u_vec - T * s0_vec))
    
    return A_gas_total + A_solid

@jax.jit
def compute_chemical_potential_jcz3(
    n: jnp.ndarray,
    V: float,
    T: float,
    coeffs_all: jnp.ndarray,
    epsilon_matrix: jnp.ndarray,
    r_star_matrix: jnp.ndarray,
    alpha_matrix: jnp.ndarray,
    solid_mask: Optional[jnp.ndarray] = None,
    solid_v0: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    grad_fn = jax.grad(compute_total_helmholtz_energy, argnums=0)
    mu_vec = grad_fn(n, V, T, coeffs_all, epsilon_matrix, r_star_matrix, alpha_matrix, solid_mask, solid_v0)
    return mu_vec

@jax.jit
def compute_pressure_jcz3(
    n: jnp.ndarray,
    V: float,
    T: float,
    epsilon_matrix: jnp.ndarray,
    r_star_matrix: jnp.ndarray,
    alpha_matrix: jnp.ndarray,
    solid_mask: Optional[jnp.ndarray] = None,
    solid_v0: Optional[jnp.ndarray] = None
) -> float:
    coeffs_dummy = jnp.zeros((n.shape[0], 7))
    grad_fn = jax.grad(compute_total_helmholtz_energy, argnums=1)
    dA_dV = grad_fn(n, V, T, coeffs_dummy, epsilon_matrix, r_star_matrix, alpha_matrix, solid_mask, solid_v0) 
    return -dA_dV * 1e6

@jax.jit
def compute_internal_energy_jcz3(
    n: jnp.ndarray,
    V: float,
    T: float,
    coeffs_all: jnp.ndarray,
    epsilon_matrix: jnp.ndarray,
    r_star_matrix: jnp.ndarray,
    alpha_matrix: jnp.ndarray,
    solid_mask: Optional[jnp.ndarray] = None,
    solid_v0: Optional[jnp.ndarray] = None
) -> float:
    grad_T_fn = jax.grad(compute_total_helmholtz_energy, argnums=2)
    dA_dT = grad_T_fn(n, V, T, coeffs_all, epsilon_matrix, r_star_matrix, alpha_matrix, solid_mask, solid_v0)
    A = compute_total_helmholtz_energy(n, V, T, coeffs_all, epsilon_matrix, r_star_matrix, alpha_matrix, solid_mask, solid_v0)
    return A - T * dA_dT

@jax.jit
def compute_entropy_consistent(
    n: jnp.ndarray,
    V: float,
    T: float,
    coeffs_all: jnp.ndarray,
    epsilon_matrix: jnp.ndarray,
    r_star_matrix: jnp.ndarray,
    alpha_matrix: jnp.ndarray,
    solid_mask: Optional[jnp.ndarray] = None,
    solid_v0: Optional[jnp.ndarray] = None
) -> float:
    def h_wrt_T(t):
        return compute_total_helmholtz_energy(n, V, t, coeffs_all, epsilon_matrix, r_star_matrix, alpha_matrix, solid_mask, solid_v0)
    return -jax.grad(h_wrt_T)(T)
