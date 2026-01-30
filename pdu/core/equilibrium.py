"""
化学平衡求解器模块 (Accelerated Schur-RAND)

基于 Schur Complement 加速的 KKT 求解器，专为 RTX 4060 等 FP32 主力硬件优化。
API 已重构为直接接受数值矩阵 (A)，避免传递字符串导致的 JAX Tracing 问题。
"""

import jax
import jax.numpy as jnp
from jax import custom_vjp, jit, lax
from typing import Tuple, NamedTuple, Optional

from pdu.utils.precision import to_fp32, to_fp64, R_GAS
from pdu.physics.thermo import compute_chemical_potential, compute_gibbs_batch
from pdu.physics.eos import compute_pressure_jcz3, compute_chemical_potential_jcz3

# ==============================================================================
# Helper for matrix building (User utility)
# ==============================================================================
def build_stoichiometry_matrix(species_list, elements):
    species_elements = {
        'N2': {'N': 2}, 'CO2': {'C': 1, 'O': 2}, 'H2O': {'H': 2, 'O': 1},
        'CO': {'C': 1, 'O': 1}, 'H2': {'H': 2}, 'O2': {'O': 2},
        'NO': {'N': 1, 'O': 1}, 'OH': {'O': 1, 'H': 1},
        'NH3': {'N': 1, 'H': 3}, 'CH4': {'C': 1, 'H': 4},
        'C_graphite': {'C': 1}, 'Al2O3': {'Al': 2, 'O': 3},
        'Al': {'Al': 1}, 'AlO': {'Al': 1, 'O': 1}, 
        'Al2O': {'Al': 2, 'O': 1}, 'AlOH': {'Al': 1, 'O': 1, 'H': 1},
        'AlO2': {'Al': 1, 'O': 2}
    }
    n_elem = len(elements)
    n_spec = len(species_list)
    A = jnp.zeros((n_elem, n_spec), dtype=jnp.float32)
    for i, s in enumerate(species_list):
        comp = species_elements.get(s, {})
        for j, e in enumerate(elements):
            A = A.at[j, i].set(float(comp.get(e, 0)))
    return A

# ==============================================================================
# 核心算法：基于 Schur 补的 KKT 求解器
# ==============================================================================

@jit
def solve_kkt_schur(
    n: jnp.ndarray,
    mu_total: jnp.ndarray,
    A: jnp.ndarray,
    residual_elem: jnp.ndarray,
    active_mask: jnp.ndarray,
    T: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Schur Complement KKT Solver"""
    n_safe = jnp.maximum(n, 1e-20)
    H_inv_diag = to_fp32(n_safe / (R_GAS * T)) * active_mask
    
    M = (A * H_inv_diag) @ A.T
    
    grad_G = to_fp32(mu_total)
    rhs = to_fp64(residual_elem) - to_fp64(A @ (H_inv_diag * grad_G))
    
    M_64 = to_fp64(M)
    M_reg = M_64 + 1e-9 * jnp.eye(M.shape[0], dtype=jnp.float64)
    lambda_new = jax.scipy.linalg.solve(M_reg, rhs)
    
    grad_total = grad_G + to_fp32(A.T @ lambda_new)
    dn = -H_inv_diag * grad_total
    
    return dn, lambda_new


@jit
def _equilibrium_loop_body(state, params_static):
    """求解循环体 (V8支持多相)"""
    n, A, b_elem, coeffs, V, T, active_mask, damping, iter_idx, eos_params = state
    # eps, r_star, alpha, lambda, [solid_mask, solid_v0], [n_fixed, v0_fixed, e_fixed]
    eps, r_star, alpha = eos_params[:3]
    lamb = eos_params[3] if len(eos_params) > 3 else None
    
    solid_mask = eos_params[4] if len(eos_params) > 4 else None
    solid_v0 = eos_params[5] if len(eos_params) > 5 else None
    
    # V9 Params
    n_fixed = eos_params[6] if len(eos_params) > 6 else 0.0
    v0_fixed = eos_params[7] if len(eos_params) > 7 else 10.0
    e_fixed = eos_params[8] if len(eos_params) > 8 else 0.0
    r_corr = eos_params[9] if len(eos_params) > 9 else 0.0
    
    # 1. 计算化学势 (JCZ3 AutoDiff)
    mu = compute_chemical_potential_jcz3(n, V, T, coeffs, eps, r_star, alpha, lamb, solid_mask, solid_v0, n_fixed, v0_fixed, e_fixed, r_corr)
    
    # 2. 计算残差
    n_64 = to_fp64(n)
    res_elem = to_fp64(A) @ n_64 - to_fp64(b_elem)
    
    # 3. KKT 步
    dn, lambda_new = solve_kkt_schur(
        n, mu, A, res_elem, active_mask, T
    )
    
    # 4. 更新
    def apply_update(ni, dni):
        factor = jnp.where(dni < 0, 0.9 * ni / (jnp.abs(dni) + 1e-30), 1.0)
        # Add protection against runaway updates
        factor = jnp.minimum(factor, 1.0)
        new_ni = ni + dni * factor * damping
        new_ni = jnp.maximum(new_ni, 1e-25)
        # Handle nans explicitly to prevent poisoning
        new_ni = jnp.nan_to_num(new_ni, nan=1e-25, posinf=1.0, neginf=1e-25)
        return new_ni
        
    n_new = apply_update(n, dn)
    n_new = jnp.maximum(n_new, 1e-20)
    new_res_norm = jnp.linalg.norm(res_elem)
    
    return (n_new, A, b_elem, coeffs, V, T, active_mask, damping, iter_idx + 1, eos_params), new_res_norm


def _solve_equilibrium_impl(
    atom_vec: jnp.ndarray,
    V: float,
    T: float,
    A: jnp.ndarray,
    coeffs_all: jnp.ndarray,
    eos_params: tuple,
    n_init: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    求解化学平衡 (Primal Implementation)
    支持 Warm-Start (n_init) 以加速 ODE 空间步积分。
    """
    n_species = A.shape[1]
    if n_init is None:
        # 质量一致性初始猜想: A @ n = atom_vec
        # 简单分配到各物种
        n_start = jnp.ones(n_species) * (jnp.sum(atom_vec) / (jnp.sum(A) + 1e-10))
    else:
        n_start = n_init
        
    active_mask = jnp.ones(n_species)
    
    # Pass eos_params into state
    init_state = (n_start, A, atom_vec, coeffs_all, V, T, active_mask, 1.0, 0, eos_params)
    
    def cond_fn(val):
        state, res_norm = val
        iter_idx = state[-2] 
        # Warm-Start 下通常只需极少迭代
        return (res_norm > 1e-4) & (iter_idx < 300)
    
    def body_fn(val):
        state, _ = val
        new_state, new_res = _equilibrium_loop_body(state, None)
        return new_state, new_res
        
    final_val = lax.while_loop(cond_fn, body_fn, (init_state, 10.0))
    n_final = final_val[0][0]
    return n_final


def solve_equilibrium_fwd(atom_vec, V, T, A, coeffs_all, eos_params, n_init=None):
    n_star = _solve_equilibrium_impl(atom_vec, V, T, A, coeffs_all, eos_params, n_init)
    return n_star, (n_star, atom_vec, V, T, A, coeffs_all, eos_params)

def solve_equilibrium_bwd(res, g_n):
    n_star, atom_vec, V, T, A, coeffs, eos_params = res
    eps, r_s, alpha = eos_params[:3]
    lamb = eos_params[3] if len(eos_params) > 3 else None
    solid_mask = eos_params[4] if len(eos_params) > 4 else None
    solid_v0 = eos_params[5] if len(eos_params) > 5 else None
    n_fixed = eos_params[6] if len(eos_params) > 6 else 0.0
    v0_fixed = eos_params[7] if len(eos_params) > 7 else 10.0
    e_fixed = eos_params[8] if len(eos_params) > 8 else 0.0
    r_corr = eos_params[9] if len(eos_params) > 9 else 0.0
    
    from pdu.physics.eos import compute_total_helmholtz_energy
    H_exact = jax.hessian(compute_total_helmholtz_energy, argnums=0)(
        n_star, V, T, coeffs, eps, r_s, alpha, lamb, solid_mask, solid_v0, n_fixed, v0_fixed, e_fixed, r_corr
    )
    
    H_inv = jax.scipy.linalg.inv(H_exact + 1e-6 * jnp.eye(H_exact.shape[0]))
    M = A @ H_inv @ A.T
    M_reg = to_fp64(M) + 1e-9 * jnp.eye(M.shape[0], dtype=jnp.float64)
    rhs_schur = A @ (H_inv @ g_n)
    lambda_adj = jax.scipy.linalg.solve(M_reg, to_fp64(rhs_schur))
    
    term = g_n - A.T @ lambda_adj
    z_n = H_inv @ term
    
    grad_atom = lambda_adj
    grad_V = None 
    grad_T = None 
    grad_A = None 
    grad_coeffs = None
    
    from pdu.physics.eos import compute_chemical_potential_jcz3
    
    # Sensitivity w.r.t EOS Params (eps, r, alpha, solid_mask, solid_v0)
    # We maintain the structure of eos_params
    def computed_mu_for_diff(*params):
        # params matches eos_params structure
        e, r, a = params[:3]
        lam = params[3] if len(params) > 3 else None
        sm = params[4] if len(params) > 4 else None
        sv = params[5] if len(params) > 5 else None
        nf = params[6] if len(params) > 6 else 0.0
        vf = params[7] if len(params) > 7 else 10.0
        ef = params[8] if len(params) > 8 else 0.0
        rc = params[9] if len(params) > 9 else 0.0
        return compute_chemical_potential_jcz3(n_star, V, T, coeffs, e, r, a, lam, sm, sv, nf, vf, ef, rc)
    
    _, vjp_fun = jax.vjp(computed_mu_for_diff, *eos_params)
    
    grad_eos = vjp_fun(-z_n)
    
    return grad_atom, grad_V, grad_T, grad_A, grad_coeffs, grad_eos


# nondiff_argnums: None now (allow everything to catch gradients)
solve_equilibrium = custom_vjp(_solve_equilibrium_impl)
solve_equilibrium.defvjp(solve_equilibrium_fwd, solve_equilibrium_bwd)
