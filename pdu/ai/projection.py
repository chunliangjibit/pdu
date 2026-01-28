"""
原子守恒投影层模块

将 MLP 预测的初值投影到满足原子守恒约束的可行域。
确保：A @ n = b，其中 A 是化学计量矩阵，b 是原子总量。
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from functools import partial


@jax.jit
def project_to_conservation(
    n_pred: jnp.ndarray,
    b: jnp.ndarray,
    A: jnp.ndarray,
    active_mask: jnp.ndarray = None
) -> jnp.ndarray:
    """将预测摩尔数投影到原子守恒约束
    
    求解最小二乘问题:
    min ||n - n_pred||² s.t. A @ n = b, n >= 0
    
    使用拉格朗日乘子法的解析解。
    
    Args:
        n_pred: MLP 预测的摩尔数 (n_species,)
        b: 原子总量向量 (n_elements,)
        A: 化学计量矩阵 (n_elements, n_species)
        active_mask: 活性物种掩码，可选
        
    Returns:
        满足约束的摩尔数 n_proj
    """
    if active_mask is None:
        active_mask = jnp.ones(n_pred.shape[0])
    
    # 应用掩码
    n_pred = n_pred * active_mask
    
    # 约束违反量
    residual = A @ n_pred - b
    
    # 投影方向: 使用伪逆
    # n_proj = n_pred - A^T @ (A @ A^T)^{-1} @ residual
    AAT = A @ A.T
    # 正则化以避免奇异
    AAT_reg = AAT + 1e-6 * jnp.eye(AAT.shape[0])
    
    lambda_opt = jax.scipy.linalg.solve(AAT_reg, residual)
    correction = A.T @ lambda_opt
    
    n_proj = n_pred - correction
    
    # 非负约束
    n_proj = jnp.maximum(n_proj, 0.0) * active_mask
    
    return n_proj


@jax.jit
def project_log_space(
    z_pred: jnp.ndarray,
    b: jnp.ndarray,
    A: jnp.ndarray,
    active_mask: jnp.ndarray = None,
    n_iters: int = 5
) -> jnp.ndarray:
    """对数域投影
    
    在对数域进行迭代投影，更好地处理数值范围。
    
    Args:
        z_pred: 对数摩尔数预测 ln(n)
        b: 原子总量
        A: 化学计量矩阵
        active_mask: 活性掩码
        n_iters: 迭代次数
        
    Returns:
        投影后的对数摩尔数 z_proj
    """
    if active_mask is None:
        active_mask = jnp.ones(z_pred.shape[0])
    
    z = z_pred
    
    for _ in range(n_iters):
        # 转换到线性空间
        n = jnp.exp(z) * active_mask
        
        # 投影
        n_proj = project_to_conservation(n, b, A, active_mask)
        
        # 转回对数空间
        z = jnp.log(n_proj + 1e-30)
        z = jnp.clip(z, -50.0, 10.0)
    
    return z


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def project_with_gradient(
    n_pred: jnp.ndarray,
    b: jnp.ndarray,
    A: jnp.ndarray,
    active_mask: jnp.ndarray
) -> jnp.ndarray:
    """带梯度的投影层
    
    使用 custom_vjp 确保梯度正确传播。
    """
    return project_to_conservation(n_pred, b, A, active_mask)


def project_with_gradient_fwd(n_pred, b, A, active_mask):
    """前向传播"""
    n_proj = project_to_conservation(n_pred, b, A, active_mask)
    return n_proj, (n_pred, b, A, active_mask, n_proj)


def project_with_gradient_bwd(A, active_mask, res, g):
    """反向传播
    
    投影层的梯度：
    dn_proj/dn_pred = I - A^T @ (A @ A^T)^{-1} @ A
    """
    n_pred, b, _, _, n_proj = res
    
    # 投影矩阵的梯度
    AAT = A @ A.T + 1e-6 * jnp.eye(A.shape[0])
    AAT_inv = jnp.linalg.inv(AAT)
    P = jnp.eye(A.shape[1]) - A.T @ AAT_inv @ A
    
    # 梯度传播
    g_n_pred = P @ g
    g_b = AAT_inv @ (A @ g)
    
    return (g_n_pred, g_b)


project_with_gradient.defvjp(project_with_gradient_fwd, project_with_gradient_bwd)


class ConservationProjector:
    """原子守恒投影器
    
    封装投影操作，缓存矩阵分解以提高效率。
    """
    
    def __init__(
        self,
        species_list: list,
        elements: list = None
    ):
        """初始化投影器
        
        Args:
            species_list: 物种名称列表
            elements: 元素列表
        """
        self.species_list = species_list
        self.elements = elements or ['C', 'H', 'N', 'O', 'Cl', 'Al', 'Mg', 'B', 'K']
        
        # 构建化学计量矩阵
        self.A = self._build_stoichiometry_matrix()
        self._AAT_inv = None
        
    def _build_stoichiometry_matrix(self) -> jnp.ndarray:
        """构建化学计量矩阵"""
        species_elements = {
            'N2': {'N': 2},
            'CO2': {'C': 1, 'O': 2},
            'H2O': {'H': 2, 'O': 1},
            'CO': {'C': 1, 'O': 1},
            'H2': {'H': 2},
            'O2': {'O': 2},
            'NO': {'N': 1, 'O': 1},
            'NO2': {'N': 1, 'O': 2},
            'OH': {'O': 1, 'H': 1},
            'CH4': {'C': 1, 'H': 4},
            'NH3': {'N': 1, 'H': 3},
            'HCN': {'H': 1, 'C': 1, 'N': 1},
            'HCl': {'H': 1, 'Cl': 1},
            'Cl2': {'Cl': 2},
            'C_graphite': {'C': 1},
            'Al2O3': {'Al': 2, 'O': 3},
            'AlO': {'Al': 1, 'O': 1},
            'MgO': {'Mg': 1, 'O': 1},
            'B2O3': {'B': 2, 'O': 3},
        }
        
        n_elem = len(self.elements)
        n_spec = len(self.species_list)
        
        A = jnp.zeros((n_elem, n_spec))
        
        for i, species in enumerate(self.species_list):
            elem_dict = species_elements.get(species, {})
            for j, elem in enumerate(self.elements):
                A = A.at[j, i].set(float(elem_dict.get(elem, 0)))
        
        return A
    
    def project(
        self,
        n_pred: jnp.ndarray,
        atom_vector: jnp.ndarray,
        active_mask: jnp.ndarray = None
    ) -> jnp.ndarray:
        """执行投影
        
        Args:
            n_pred: 预测摩尔数
            atom_vector: 原子总量 (mol)
            active_mask: 活性掩码
            
        Returns:
            投影后的摩尔数
        """
        if active_mask is None:
            active_mask = jnp.ones(len(self.species_list))
        
        return project_to_conservation(n_pred, atom_vector, self.A, active_mask)
    
    def project_with_grad(
        self,
        n_pred: jnp.ndarray,
        atom_vector: jnp.ndarray,
        active_mask: jnp.ndarray = None
    ) -> jnp.ndarray:
        """带梯度的投影
        
        Args:
            n_pred: 预测摩尔数
            atom_vector: 原子总量
            active_mask: 活性掩码
            
        Returns:
            投影后的摩尔数（支持梯度传播）
        """
        if active_mask is None:
            active_mask = jnp.ones(len(self.species_list))
        
        return project_with_gradient(n_pred, atom_vector, self.A, active_mask)
    
    def check_conservation(
        self,
        n: jnp.ndarray,
        atom_vector: jnp.ndarray,
        tol: float = 1e-6
    ) -> Tuple[bool, jnp.ndarray]:
        """检查原子守恒
        
        Args:
            n: 摩尔数
            atom_vector: 原子总量
            tol: 容差
            
        Returns:
            (是否满足, 残差向量)
        """
        residual = self.A @ n - atom_vector
        satisfied = jnp.all(jnp.abs(residual) < tol)
        return bool(satisfied), residual
