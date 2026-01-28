"""
Exp-6 分子势能函数模块

实现 JCZ3 状态方程使用的 Exp-6 势能函数。
遵循混合精度策略：势能计算使用 FP32 加速。
"""

import jax
import jax.numpy as jnp
from typing import Tuple

from pdu.utils.precision import to_fp32, to_fp64


@jax.jit
def exp6_potential(
    r: float,
    epsilon: float,
    r_star: float,
    alpha: float
) -> float:
    """Exp-6 势能函数
    
    公式:
    φ(r) = ε/(1-6/α) * [6/α * exp(α*(1-r/r*)) - (r*/r)^6]
    
    Args:
        r: 分子间距离 (Å)
        epsilon: 势阱深度 (K)，即 ε/k_B
        r_star: 平衡距离 (Å)
        alpha: 硬度参数 (无量纲)
        
    Returns:
        势能 φ (K)，以温度单位表示
    """
    # 使用 FP32 进行计算加速
    r = to_fp32(jnp.asarray(r))
    epsilon = to_fp32(jnp.asarray(epsilon))
    r_star = to_fp32(jnp.asarray(r_star))
    alpha = to_fp32(jnp.asarray(alpha))
    
    # 防止除零
    r = jnp.maximum(r, 0.1)
    
    # 预计算因子
    ratio = r_star / r
    exp_term = jnp.exp(alpha * (1.0 - r / r_star))
    
    prefactor = epsilon / (1.0 - 6.0 / alpha)
    
    phi = prefactor * (6.0 / alpha * exp_term - ratio**6)
    
    return to_fp64(phi)


@jax.jit
def exp6_potential_derivative(
    r: float,
    epsilon: float,
    r_star: float,
    alpha: float
) -> Tuple[float, float]:
    """Exp-6 势能及其导数
    
    同时计算势能和力 (dφ/dr)。
    
    Args:
        r: 分子间距离 (Å)
        epsilon: 势阱深度 (K)
        r_star: 平衡距离 (Å)
        alpha: 硬度参数
        
    Returns:
        (φ, dφ/dr) 元组
    """
    # 使用自动微分计算导数
    phi_func = lambda r_: exp6_potential(r_, epsilon, r_star, alpha)
    
    phi = phi_func(r)
    dphi_dr = jax.grad(phi_func)(r)
    
    return phi, dphi_dr


@jax.jit
def mixing_rule_lorentz_berthelot(
    eps1: float, r1: float, alpha1: float,
    eps2: float, r2: float, alpha2: float
) -> Tuple[float, float, float]:
    """Lorentz-Berthelot 混合规则
    
    用于计算不同物种间的交叉势能参数。
    
    ε_12 = √(ε_1 * ε_2)
    r*_12 = (r*_1 + r*_2) / 2
    α_12 = (α_1 + α_2) / 2
    
    Args:
        eps1, r1, alpha1: 物种1的参数
        eps2, r2, alpha2: 物种2的参数
        
    Returns:
        (ε_12, r*_12, α_12) 元组
    """
    eps12 = jnp.sqrt(eps1 * eps2)
    r12 = (r1 + r2) / 2.0
    alpha12 = (alpha1 + alpha2) / 2.0
    
    return eps12, r12, alpha12


@jax.jit
def compute_pair_potential_matrix(
    r: float,
    epsilon_matrix: jnp.ndarray,
    r_star_matrix: jnp.ndarray,
    alpha_matrix: jnp.ndarray
) -> jnp.ndarray:
    """计算所有物种对的势能矩阵
    
    Args:
        r: 分子间距离 (Å)
        epsilon_matrix: ε_ij 矩阵 (n, n)
        r_star_matrix: r*_ij 矩阵 (n, n)
        alpha_matrix: α_ij 矩阵 (n, n)
        
    Returns:
        势能矩阵 φ_ij (n, n) (K)
    """
    # 向量化计算
    def compute_phi(eps, r_star, alpha):
        return exp6_potential(r, eps, r_star, alpha)
    
    # 使用 vmap 双重向量化
    phi_matrix = jax.vmap(
        jax.vmap(compute_phi, in_axes=(0, 0, 0)), 
        in_axes=(0, 0, 0)
    )(epsilon_matrix, r_star_matrix, alpha_matrix)
    
    return phi_matrix


def build_mixing_matrices(
    species_params: dict
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """构建混合规则参数矩阵
    
    Args:
        species_params: 物种参数字典 {name: (eps, r_star, alpha)}
        
    Returns:
        (epsilon_matrix, r_star_matrix, alpha_matrix) 元组
    """
    species_list = list(species_params.keys())
    n = len(species_list)
    
    eps_matrix = jnp.zeros((n, n))
    r_matrix = jnp.zeros((n, n))
    alpha_matrix = jnp.zeros((n, n))
    
    for i, si in enumerate(species_list):
        eps_i, r_i, alpha_i = species_params[si]
        for j, sj in enumerate(species_list):
            eps_j, r_j, alpha_j = species_params[sj]
            
            eps_ij, r_ij, alpha_ij = mixing_rule_lorentz_berthelot(
                eps_i, r_i, alpha_i, eps_j, r_j, alpha_j
            )
            
            eps_matrix = eps_matrix.at[i, j].set(eps_ij)
            r_matrix = r_matrix.at[i, j].set(r_ij)
            alpha_matrix = alpha_matrix.at[i, j].set(alpha_ij)
    
    return eps_matrix, r_matrix, alpha_matrix


@jax.jit
def compute_second_virial_exp6(
    T: float,
    epsilon: float,
    r_star: float,
    alpha: float,
    n_points: int = 100
) -> float:
    """计算 Exp-6 势能的第二维里系数 B(T)
    
    B(T) = 2π N_A ∫[1 - exp(-φ(r)/kT)] r² dr
    
    使用数值积分。
    
    Args:
        T: 温度 (K)
        epsilon: 势阱深度 (K)
        r_star: 平衡距离 (Å)
        alpha: 硬度参数
        n_points: 积分点数
        
    Returns:
        B(T) (cm³/mol)
    """
    # 积分范围: 0.5*r_star 到 5*r_star
    r_min = 0.5 * r_star
    r_max = 5.0 * r_star
    
    r_vals = jnp.linspace(r_min, r_max, n_points)
    dr = (r_max - r_min) / (n_points - 1)
    
    def integrand(r):
        phi = exp6_potential(r, epsilon, r_star, alpha)
        # φ 已经以 K 为单位
        boltzmann = jnp.exp(-phi / T)
        return (1.0 - boltzmann) * r**2
    
    integrand_vals = jax.vmap(integrand)(r_vals)
    
    # 梯形积分
    integral = jnp.trapezoid(integrand_vals, r_vals)
    
    # 阿伏伽德罗常数
    N_A = 6.02214076e23
    
    # 转换: Å³ -> cm³ (1 Å³ = 1e-24 cm³)
    B = 2.0 * jnp.pi * N_A * integral * 1e-24
    
    return B
