"""
混合精度牛顿法模块

实现 FP32/FP64 混合精度的牛顿迭代求解器。
设计原则：
- FP32 计算方向（Jacobian 矩阵和线性求解）
- FP64 计算残差和状态更新
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable, Tuple, NamedTuple
from functools import partial

from pdu.utils.precision import to_fp32, to_fp64


class NewtonState(NamedTuple):
    """牛顿迭代状态"""
    x: jnp.ndarray          # 当前解 (FP64)
    residual: jnp.ndarray   # 残差向量 (FP64)
    iteration: int          # 迭代次数
    converged: bool         # 是否收敛


class NewtonConfig(NamedTuple):
    """牛顿法配置"""
    max_iter: int = 50          # 最大迭代次数
    rtol: float = 1e-8          # 相对容差
    atol: float = 1e-10         # 绝对容差
    damping: float = 1.0        # 阻尼因子
    use_line_search: bool = False  # 是否使用线搜索


def _check_convergence(
    residual: jnp.ndarray,
    residual_prev: jnp.ndarray,
    rtol: float,
    atol: float
) -> bool:
    """检查收敛性
    
    Args:
        residual: 当前残差
        residual_prev: 上一步残差
        rtol: 相对容差
        atol: 绝对容差
        
    Returns:
        是否收敛
    """
    residual = to_fp64(residual)
    
    # 残差范数
    res_norm = jnp.linalg.norm(residual)
    
    # 绝对收敛判断
    abs_converged = res_norm < atol
    
    # 相对收敛判断
    res_prev_norm = jnp.linalg.norm(residual_prev)
    rel_converged = res_norm < rtol * jnp.maximum(res_prev_norm, 1.0)
    
    return abs_converged | rel_converged


@partial(jax.jit, static_argnums=(0, 1))
def mixed_precision_newton(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    jacobian_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    config: NewtonConfig = NewtonConfig()
) -> NewtonState:
    """混合精度牛顿法求解器
    
    求解 F(x) = 0
    
    使用混合精度策略:
    - Jacobian 矩阵计算和线性求解使用 FP32
    - 残差计算和状态更新使用 FP64
    
    Args:
        residual_fn: 残差函数 F(x)
        jacobian_fn: Jacobian 矩阵函数 J(x) = dF/dx
        x0: 初始猜测值
        config: 牛顿法配置
        
    Returns:
        NewtonState 包含最终解和收敛信息
    """
    x0 = to_fp64(x0)
    
    def newton_step(carry, _):
        x, residual_prev, converged = carry
        
        # 计算残差 (FP64)
        residual = to_fp64(residual_fn(x))
        
        # 计算 Jacobian (FP32 加速)
        J = to_fp32(jacobian_fn(x))
        
        # 求解线性系统 J * delta = -residual (FP32)
        residual_fp32 = to_fp32(residual)
        
        # 使用 LU 分解求解
        # delta = -J^{-1} * residual
        try:
            delta = jax.scipy.linalg.solve(J, -residual_fp32)
        except:
            # 如果矩阵奇异，使用最小二乘
            delta = jnp.linalg.lstsq(J, -residual_fp32, rcond=None)[0]
        
        # 状态更新 (FP64)
        delta = to_fp64(delta)
        x_new = x + config.damping * delta
        
        # 检查收敛
        converged_new = _check_convergence(residual, residual_prev, config.rtol, config.atol)
        
        return (x_new, residual, converged | converged_new), None
    
    # 初始状态
    residual_init = to_fp64(residual_fn(x0))
    init_state = (x0, residual_init, False)
    
    # 迭代
    (x_final, residual_final, converged), _ = lax.scan(
        newton_step, 
        init_state, 
        None, 
        length=config.max_iter
    )
    
    return NewtonState(
        x=x_final,
        residual=residual_final,
        iteration=config.max_iter,
        converged=converged
    )


@partial(jax.jit, static_argnums=(0,))
def newton_with_autodiff_jacobian(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    config: NewtonConfig = NewtonConfig()
) -> NewtonState:
    """使用自动微分计算 Jacobian 的牛顿法
    
    适用于残差函数可微分的情况。
    
    Args:
        residual_fn: 残差函数 F(x)
        x0: 初始猜测值
        config: 牛顿法配置
        
    Returns:
        NewtonState
    """
    # 自动微分计算 Jacobian
    jacobian_fn = jax.jacfwd(residual_fn)
    
    return mixed_precision_newton(residual_fn, jacobian_fn, x0, config)


@jax.jit
def newton_step_single(
    x: jnp.ndarray,
    residual_fn: Callable,
    jacobian_fn: Callable,
    damping: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """单步牛顿迭代
    
    用于需要手动控制迭代的场景。
    
    Args:
        x: 当前解
        residual_fn: 残差函数
        jacobian_fn: Jacobian 函数
        damping: 阻尼因子
        
    Returns:
        (x_new, residual) 元组
    """
    x = to_fp64(x)
    
    # 计算残差和 Jacobian
    residual = to_fp64(residual_fn(x))
    J = to_fp32(jacobian_fn(x))
    
    # 求解
    delta = to_fp64(jax.scipy.linalg.solve(J, -to_fp32(residual)))
    
    # 更新
    x_new = x + damping * delta
    
    return x_new, residual


def newton_with_line_search(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    jacobian_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
    alpha_init: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5
) -> NewtonState:
    """带线搜索的牛顿法
    
    使用 Armijo 回溯线搜索确定步长。
    
    Args:
        residual_fn: 残差函数
        jacobian_fn: Jacobian 函数
        x0: 初始猜测
        max_iter: 最大迭代次数
        tol: 收敛容差
        alpha_init: 初始步长
        c1: Armijo 条件参数
        rho: 步长收缩因子
        
    Returns:
        NewtonState
    """
    x = to_fp64(x0)
    
    def merit_function(x):
        r = residual_fn(x)
        return 0.5 * jnp.sum(r ** 2)
    
    for iteration in range(max_iter):
        residual = to_fp64(residual_fn(x))
        res_norm = jnp.linalg.norm(residual)
        
        if res_norm < tol:
            return NewtonState(x=x, residual=residual, iteration=iteration, converged=True)
        
        # 计算搜索方向
        J = to_fp32(jacobian_fn(x))
        delta = to_fp64(jax.scipy.linalg.solve(J, -to_fp32(residual)))
        
        # 线搜索
        alpha = alpha_init
        merit_x = merit_function(x)
        grad_merit = jnp.dot(residual, jax.jvp(residual_fn, (x,), (delta,))[1])
        
        for _ in range(20):  # 最多 20 次回溯
            x_new = x + alpha * delta
            merit_new = merit_function(x_new)
            
            if merit_new <= merit_x + c1 * alpha * grad_merit:
                break
            alpha *= rho
        
        x = x_new
    
    final_residual = to_fp64(residual_fn(x))
    converged = jnp.linalg.norm(final_residual) < tol
    
    return NewtonState(x=x, residual=final_residual, iteration=max_iter, converged=converged)
