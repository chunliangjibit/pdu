# pdu/flux/heat.py
import jax
import jax.numpy as jnp
import equinox as eqx
from pdu.core.types import GasState, ParticleState

"""
PDU V11.0 Tier 1: 物理通量 - 传热模型
实现 Ranz-Marshall 关联式，描述气固两相间的能量交换。
"""

class RanzMarshallHeat(eqx.Module):
    multiplier: jnp.ndarray = eqx.field(converter=jnp.asarray)
    
    def __init__(self, init_val=1.0):
        self.multiplier = jnp.array(init_val)

    def __call__(self, gas: GasState, part: ParticleState) -> jnp.ndarray:
        """
        计算气固相间的传热速率 (Energy per unit volume per unit time)
        """
        # 1. 计算相对速度和雷诺数 (重复逻辑，后续可优化)
        u_rel = gas.u - part.u
        abs_u_rel = jnp.abs(u_rel)
        mu_gas = 1.0e-4 * (gas.T / 3000.0)**0.7
        rho_g_si = gas.rho * 1000.0
        Re = jnp.maximum(rho_g_si * abs_u_rel * (2.0 * part.r) / (mu_gas + 1e-10), 1e-3)
        
        # 2. 气体热物理参数
        Pr = 0.7  # 普朗特数 (典型气体)
        kappa_g = 0.2 * (gas.T / 3000.0)**0.8 # 导热系数 (W/m*K)
        
        # 3. Ranz-Marshall 关联式
        # Nu = 2 + 0.6 * Re^0.5 * Pr^(1/3)
        Nu = 2.0 + 0.6 * jnp.sqrt(Re) * (Pr ** (1.0/3.0))
        
        # 4. 对流换热系数 h = Nu * kappa / d
        h = Nu * kappa_g / (2.0 * part.r + 1e-12)
        
        # 5. 总换热速率计算
        # Q_vol = phi_p * (3 * h / r) * (T_g - T_p)
        heat_flux = part.phi * (3.0 * h / (part.r + 1e-12)) * (gas.T - part.T)
        
        return heat_flux * self.multiplier

# 向量化算子
compute_heat_bins = jax.vmap(RanzMarshallHeat(), in_axes=(None, 0))
