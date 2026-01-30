# pdu/core/types.py
import jax.numpy as jnp
from typing import NamedTuple, Optional

"""
PDU V11.0 核心数据结构 (SoA - Structure of Arrays)
旨在利用 JAX 的向量化并行能力，并最大化显存效率。
"""

class GasState(NamedTuple):
    """
    气相状态 (N_grid 点)
    """
    rho: jnp.ndarray  # 密度 (g/cm^3)
    u:   jnp.ndarray  # 速度 (m/s)
    T:   jnp.ndarray  # 温度 (K)
    lam: jnp.ndarray  # 反应进度 (0 -> 1)

class ParticleState(NamedTuple):
    """
    颗粒相状态 (N_bins x N_grid)
    利用 vmap 在 N_bins 维度进行并行计算
    """
    phi: jnp.ndarray  # 体积分数 (dimensionless)
    rho: jnp.ndarray  # 颗粒材料密度 (g/cm^3)
    u:   jnp.ndarray  # 颗粒速度 (m/s)
    T:   jnp.ndarray  # 颗粒温度 (K)
    r:   jnp.ndarray  # 瞬时半径 (m)

class State(NamedTuple):
    """
    全场综合状态 (对独立变量 x 进行积分)
    """
    gas:  GasState
    part: ParticleState

class Aux(NamedTuple):
    """
    派生物理量 (辅助计算，不参与 ODE 状态步进)
    """
    P_gas: jnp.ndarray  # 气相静压 (GPa)
    P_tot: jnp.ndarray  # 总压力 (含动量流修正) (GPa)
    cs:    jnp.ndarray  # 气相声速 (m/s)
    M_f:   jnp.ndarray  # 冻结马赫数
    u_rel: jnp.ndarray  # 相对速度 (u_g - u_p) (m/s)