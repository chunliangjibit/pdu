"""
精度控制工具模块

提供 FP32/FP64 混合精度转换和包装器。
"""

import jax
import jax.numpy as jnp
from functools import wraps
from typing import Callable, Any


def to_fp32(x: jnp.ndarray) -> jnp.ndarray:
    """将数组转换为 float32
    
    Args:
        x: 输入数组
        
    Returns:
        float32 数组
    """
    return x.astype(jnp.float32)


def to_fp64(x: jnp.ndarray) -> jnp.ndarray:
    """将数组转换为 float64
    
    Args:
        x: 输入数组
        
    Returns:
        float64 数组
    """
    return x.astype(jnp.float64)


def mixed_precision_wrapper(compute_in_fp32: bool = True):
    """混合精度包装器装饰器
    
    对于计算密集型操作，使用 FP32 加速计算，
    然后将结果转换回 FP64 以保证数值精度。
    
    Args:
        compute_in_fp32: 是否在 FP32 中执行计算
        
    Returns:
        装饰器函数
        
    Example:
        @mixed_precision_wrapper(compute_in_fp32=True)
        def heavy_computation(x, y):
            return complex_matrix_operation(x, y)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if compute_in_fp32:
                # 转换所有数组参数为 FP32
                args_fp32 = tuple(
                    to_fp32(arg) if isinstance(arg, jnp.ndarray) else arg 
                    for arg in args
                )
                kwargs_fp32 = {
                    k: to_fp32(v) if isinstance(v, jnp.ndarray) else v 
                    for k, v in kwargs.items()
                }
                
                # 执行计算
                result = func(*args_fp32, **kwargs_fp32)
                
                # 结果转回 FP64
                if isinstance(result, jnp.ndarray):
                    return to_fp64(result)
                elif isinstance(result, tuple):
                    return tuple(
                        to_fp64(r) if isinstance(r, jnp.ndarray) else r 
                        for r in result
                    )
                else:
                    return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def ensure_fp64(*names: str):
    """确保指定参数为 FP64 的装饰器
    
    用于关键的残差计算和累加操作。
    
    Args:
        *names: 需要转换为 FP64 的参数名
        
    Example:
        @ensure_fp64('residual', 'state')
        def update_step(residual, state, direction):
            return state - direction
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            
            # 转换位置参数
            args_list = list(args)
            for i, (arg, pname) in enumerate(zip(args_list, param_names)):
                if pname in names and isinstance(arg, jnp.ndarray):
                    args_list[i] = to_fp64(arg)
            
            # 转换关键字参数
            for name in names:
                if name in kwargs and isinstance(kwargs[name], jnp.ndarray):
                    kwargs[name] = to_fp64(kwargs[name])
            
            return func(*tuple(args_list), **kwargs)
        return wrapper
    return decorator


# 常量定义
R_GAS = 8.314462618  # J/(mol·K) - 通用气体常数
R_GAS_CAL = 1.987204  # cal/(mol·K)
AVOGADRO = 6.02214076e23  # 1/mol
BOLTZMANN = 1.380649e-23  # J/K
