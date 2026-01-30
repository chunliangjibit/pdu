# pdu/flux/__init__.py
from .drag import IgraDrag, compute_drag_bins
from .heat import RanzMarshallHeat, compute_heat_bins
import jax.numpy as jnp

def compute_total_sources(gas, part):
    """
    计算所有物理源项的总和 (针对多分散粒径箱)
    """
    # 1. 计算各粒径箱的阻力 (N_bins, N_grid)
    drag_forces = compute_drag_bins(gas, part)
    total_drag = jnp.sum(drag_forces, axis=0) # 作用于气相的总阻力
    
    # 2. 计算各粒径箱的换热 (N_bins, N_grid)
    heat_fluxes = compute_heat_bins(gas, part)
    total_heat = jnp.sum(heat_fluxes, axis=0) # 气相失去的热量
    
    return total_drag, total_heat
