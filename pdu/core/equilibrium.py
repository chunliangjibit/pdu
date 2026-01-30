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
# Helper for 5x5 SPD Solve (Bypass cuSolver, JAX-unrolled)
# ==============================================================================
def cholesky_5x5_spd(A, jitter=1e-12):
    """手写 5×5 Cholesky：A ≈ L L^T. (Expert Feedback V11.0)"""
    A = jnp.asarray(A)
    A = 0.5 * (A + A.T) # 强制对称
    A = A + jitter * jnp.eye(5, dtype=A.dtype)
    L = jnp.zeros((5, 5), dtype=A.dtype)
    # 5x5 Python loops are unrolled during JAX tracing
    for i in range(5):
        s = jnp.dot(L[i, :i], L[i, :i])
        L = L.at[i, i].set(jnp.sqrt(jnp.maximum(A[i, i] - s, 1e-25)))
        for j in range(i + 1, 5):
            s = jnp.dot(L[j, :i], L[i, :i])
            L = L.at[j, i].set((A[j, i] - s) / L[i, i])
    return L

def solve_lower_tri_5x5(L, b):
    y = jnp.zeros((5,), dtype=L.dtype)
    for i in range(5):
        s = jnp.dot(L[i, :i], y[:i])
        y = y.at[i].set((b[i] - s) / L[i, i])
    return y

def solve_upper_tri_5x5(U, b):
    x = jnp.zeros((5,), dtype=U.dtype)
    for i in range(4, -1, -1):
        s = jnp.dot(U[i, i+1:], x[i+1:])
        x = x.at[i].set((b[i] - s) / U[i, i])
    return x

@jit
def solve_spd_5x5(A, b, jitter=1e-12):
    """5x5 SPD Solver avoiding cuSolver handle creation"""
    L = cholesky_5x5_spd(A, jitter=jitter)
    y = solve_lower_tri_5x5(L, b)
    x = solve_upper_tri_5x5(L.T, y)
    return x


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


@jit
def _n_from_pi(pi, T, V, coeffs_low, coeffs_high, A):
    """
    从元素势 pi 映射到物种摩尔数 n (Tier 0: 理想混合近似)
    n_i = exp((A^T @ pi - mu_i_standard) / RT)
    """
    # 1. 计算标准化学势 G0(T) / RT
    # 这里直接调用 thermo.py 的函数获取 H-TS
    # 为了简化，我们假设 V_gas 接近 V_total
    P0 = 1e5
    # mu_std = (H - T*S) + RT * ln(P_ideal / P0)
    # 但由于 n 是变量，我们通常写成 n_i = (V/RT) * exp((A^T pi - mu_std_0)/RT)
    from pdu.physics.thermo import compute_enthalpy, compute_entropy
    
    def _get_g0(c_low, c_high):
        h = jnp.where(T > 1000.0, compute_enthalpy(c_high, T), compute_enthalpy(c_low, T))
        s = jnp.where(T > 1000.0, compute_entropy(c_high, T), compute_entropy(c_low, T))
        return (h - T * s) / (R_GAS * T)
        
    g0_over_rt = jax.vmap(_get_g0)(coeffs_low, coeffs_high)
    
    # 2. 计算 A^T @ pi
    pi_spec = A.T @ pi  # (S,)
    
    # 3. 计算 n
    v_factor = (V * 1e-6 * P0) / (R_GAS * T + 1e-10)
    # 限制指数项避免溢出 (exp(50) 足够大)
    log_val = jnp.clip(pi_spec - g0_over_rt, -50.0, 50.0)
    n = v_factor * jnp.exp(log_val)
    return jnp.maximum(n, 1e-25)

@jit
def _solve_pi_newton(atom_vec, V, T, A, coeffs_low, coeffs_high, pi_init):
    """5x5 Newton 迭代求解元素势 pi"""
    def body_fn(pi):
        n = _n_from_pi(pi, T, V, coeffs_low, coeffs_high, A)
        res = A @ n - atom_vec
        J = (A * n) @ A.T
        # V11 Phase 4: 使用 广义逆 (pinv) 以应对秩亏并提高全域收敛性
        J_reg = J + 1e-4 * jnp.eye(J.shape[0])
        dpi = - jnp.linalg.pinv(J_reg) @ res
        
        # 严格阻尼 Newton (V11 Ultra-Stable)
        step_norm = jnp.linalg.norm(dpi)
        scale = jnp.minimum(1.0, 5.0 / (step_norm + 1e-10))
        # 使用极小步长 (0.05) 以对抗超高压下的指数不稳定性
        return pi + 0.05 * scale * dpi
        
    # 为保证超高密度下的绝对收敛，使用 200 次迭代
    pi_final = lax.fori_loop(0, 200, lambda i, p: body_fn(p), pi_init)
    
    # 验证最终残差
    n_final = _n_from_pi(pi_final, T, V, coeffs_low, coeffs_high, A)
    res_final = A @ n_final - atom_vec
    jax.debug.print("Newton: res_norm={rn}, n_total={nt}", rn=jnp.linalg.norm(res_final), nt=jnp.sum(n_final))
    
    return pi_final

def _solve_equilibrium_impl(
    atom_vec: jnp.ndarray,
    V: float,
    T: float,
    A: jnp.ndarray,
    coeffs_low: jnp.ndarray,
    coeffs_high: jnp.ndarray,
    eos_params: tuple,
    n_init: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    基于元素势法 (EPM) 的化学平衡求解器 (V11.0 Tier 0)
    """
    n_species = A.shape[1]
    n_elem = A.shape[0]
    
    # 元素势初值优化：基于 atom_vec 平均浓度的启发式猜测
    # pi ≈ g0_over_rt - log(V_factor/n_avg)
    v_factor = (to_fp64(V) * 1e-6 * 1e5) / (R_GAS * to_fp64(T) + 1e-10)
    # 基于温度的初值自适应：高壓下 G0/RT 典型值在 -20 ~ -60
    # 我們選擇一個略微偏向低濃度的初值以獲得單調上升的收斂軌跡
    pi_base = jnp.where(to_fp64(T) < 1000.0, -60.0, -35.0)
    pi_start = jnp.full(n_elem, pi_base, dtype=jnp.float64)
    
    # 核心求解: 5x5 Newton
    pi_star = _solve_pi_newton(to_fp64(atom_vec), to_fp64(V), to_fp64(T), to_fp64(A), coeffs_low, coeffs_high, pi_start)
    
    # 映射回 n
    n_star = _n_from_pi(pi_star, to_fp64(T), to_fp64(V), coeffs_low, coeffs_high, to_fp64(A))
    
    return n_star



def solve_equilibrium_fwd(atom_vec, V, T, A, coeffs_low, coeffs_high, eos_params, n_init=None):
    """前向计算：保存 pi_star 以供隐式惩罚"""
    pi_start = jnp.zeros(A.shape[0])
    pi_star = _solve_pi_newton(to_fp64(atom_vec), to_fp64(V), to_fp64(T), to_fp64(A), coeffs_low, coeffs_high, pi_start)
    n_star = _n_from_pi(pi_star, to_fp64(T), to_fp64(V), coeffs_low, coeffs_high, to_fp64(A))
    return n_star, (pi_star, n_star, atom_vec, V, T, A, coeffs_low, coeffs_high, eos_params)

def solve_equilibrium_bwd(res, g_n):
    """反向计算：基于隐式函数定理 (IFT) 的 5x5 伴随求解"""
    pi_star, n_star, atom_vec, V, T, A, coeffs_low, coeffs_high, eos_params = res
    
    # 1. 计算 pi 空间的梯度映射
    _, vjp_n_pi = jax.vjp(lambda pi: _n_from_pi(pi, T, V, coeffs_low, coeffs_high, A), pi_star)
    (g_pi,) = vjp_n_pi(g_n) # (E,)
    
    # 2. 解伴随方程 (∂F/∂pi)^T v = g_pi
    J = (A * n_star) @ A.T
    v = solve_spd_5x5(J, g_pi)
    
    # 3. 计算对参数 theta 的梯度
    def n_and_F(V_in, T_in, c_low, c_high):
        n = _n_from_pi(pi_star, T_in, V_in, c_low, c_high, A)
        F = A @ n
        return n, F
        
    _, vjp_params = jax.vjp(n_and_F, V, T, coeffs_low, coeffs_high)
    (grad_V, grad_T, cl_grad, ch_grad) = vjp_params((g_n, -v))
    
    grad_eos = jax.tree_util.tree_map(jnp.zeros_like, eos_params)
    
    # 必须返回 8 个梯度以匹配 _solve_equilibrium_impl (atom_vec, V, T, A, coeffs_low, coeffs_high, eos_params, n_init)
    return v, grad_V, grad_T, None, cl_grad, ch_grad, grad_eos, None

# nondiff_argnums: None now (allow everything to catch gradients)
solve_equilibrium = jax.custom_vjp(_solve_equilibrium_impl)
solve_equilibrium.defvjp(solve_equilibrium_fwd, solve_equilibrium_bwd)


