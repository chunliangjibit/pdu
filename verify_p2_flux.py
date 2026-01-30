# verify_p2_flux.py
import jax
import jax.numpy as jnp
from pdu.core.types import GasState, ParticleState
from pdu.flux import compute_total_sources

def test_inert_slip_relaxation():
    print("Testing P2: Two-Phase Flux (Slip Relaxation)...")
    
    # 1. 初始状态：气流高速，颗粒静止 (模拟激波刚过后)
    # 假设 5 个粒径箱 (2um, 5um, 10um, 20um, 50um)
    r_bins = jnp.array([1e-6, 2.5e-6, 5e-6, 10e-6, 25e-6])
    phi_bins = jnp.ones(5) * 0.04 # 总体积分数 20%
    
    gas = GasState(
        rho=jnp.array(2.0), 
        u=jnp.array(2000.0), 
        T=jnp.array(3500.0), 
        lam=jnp.array(0.0)
    )
    
    # 将颗粒状态扩展为 (N_bins, N_grid)
    # 此处仅测试单点，故为 (5,)
    part = ParticleState(
        phi=phi_bins,
        rho=jnp.ones(5) * 2.7, # 铝
        u=jnp.zeros(5),
        T=jnp.ones(5) * 300.0,
        r=r_bins
    )
    
    # 2. 计算源项
    drag, heat = compute_total_sources(gas, part)
    
    print(f"\n--- Results for 5 Particle Bins ---")
    print(f"Gas Velocity: {gas.u} m/s, Particle Velocities: {part.u} m/s")
    print(f"Total Drag Force on Gas: {drag:.2e} N/m^3")
    print(f"Total Heat Flux to Particles: {heat:.2e} W/m^3")
    
    # 3. 验证粒径效应 (单独看各箱阻力)
    from pdu.flux import compute_drag_bins
    drag_bins = compute_drag_bins(gas, part)
    print("\nDrag per bin (N/m^3):")
    for i, r in enumerate(r_bins):
        print(f"  r = {r*1e6:4.1f} um: {drag_bins[i]:.2e}")
        
    # 小颗粒阻力应该更大 (比表面积大)
    assert drag_bins[0] > drag_bins[-1]
    print("\nFlux Particle Size Effect verification: SUCCESS")

if __name__ == "__main__":
    test_inert_slip_relaxation()
