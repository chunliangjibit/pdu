# pdu/flux/drag.py
import jax
import jax.numpy as jnp
import equinox as eqx
from pdu.core.types import GasState, ParticleState

"""
PDU V11.0 Tier 1: 物理通量 - 阻力模型
实现 Igra & Takayama 激波-颗粒阻力模型，并包含 Richardson-Zaki 稠密流修正。
"""

class IgraDrag(eqx.Module):
    # 允许通过反向传播标定的参数
    multiplier: jnp.ndarray = eqx.field(converter=jnp.asarray)
    
    def __init__(self, init_val=1.0):
        self.multiplier = jnp.array(init_val)

    def __call__(self, gas: GasState, part: ParticleState) -> jnp.ndarray:
        """
        计算颗粒对气相的阻力源项 (Force per unit volume of mixture)
        gas: 气相状态 (单一网格点或向量化)
        part: 颗粒相状态 (单一粒径箱在网格点上的状态)
        """
        # 1. 计算滑移速度 (u_g - u_p)
        u_rel = gas.u - part.u
        abs_u_rel = jnp.abs(u_rel)
        
        # 2. 估计动力学参数 (需要高压气相粘度)
        # 典型爆轰产物粘度 ~ 1e-4 Pa*s
        mu_gas = 1.0e-4 * (gas.T / 3000.0)**0.7
        
        # 雷诺数 Re = rho_g * |u_rel| * d / mu_g
        # Note: rho_g unit is g/cm3 -> kg/m3 (multiply by 1000)
        rho_g_si = gas.rho * 1000.0
        Re = jnp.maximum(rho_g_si * abs_u_rel * (2.0 * part.r) / (mu_gas + 1e-10), 1e-3)
        
        # 3. Igra & Takayama 基础阻力系数 Cd
        # Cd = (24/Re) * (1 + 0.15*Re**0.687)
        # 针对激波后的非定常修正
        Cd_base = (24.0 / Re) * (1.0 + 0.15 * Re**0.687)
        
        # 4. 稠密流修正 (Richardson-Zaki)
        # void_frac = epsilon = 1 - phi_total
        # 此处简化：计算单一粒径箱对局部的体积分数贡献
        # 在完整 ZND 中，应该传入总体积分数
        # TODO: 接入总 phi 修正
        void_frac = jnp.clip(1.0 - part.phi, 0.1, 1.0)
        dense_factor = void_frac ** (-2.65)
        
        # 5. 总阻力计算 (F = 3/8 * rho_g * Cd * dense * u_rel^2 / r)
        # 乘上体积分数以获得单位体积混合物的受力
        # Force = phi_p * (3/8 * rho_g * Cd * dense * u_rel^2 / r)
        force = part.phi * (0.375 * rho_g_si * Cd_base * dense_factor / (part.r + 1e-12)) * u_rel * abs_u_rel
        
        return force * self.multiplier

# 向量化算子：自动处理 N_BINS 维度
compute_drag_bins = jax.vmap(IgraDrag(), in_axes=(None, 0))
