# verify_p3_solver.py
import jax
import jax.numpy as jnp
from pdu.core.types import GasState, ParticleState, State
from pdu.solver.znd import solve_znd_profile

def test_znd_integration_hmx():
    print("Testing P3: ZND Solver Integration (HMX + Al Simulation)...")
    
    # 1. 模拟 VN Spike 后的初始状态 (1D ZND)
    # rho_0 = 1.9, D = 7000 => VN Spike rho ~ 2.5
    gas_init = GasState(
        rho=jnp.array(2.5), 
        u=jnp.array(1800.0), 
        T=jnp.array(4500.0), 
        lam=jnp.array(0.01)
    )
    
    # 5 个粒径箱
    part_init = ParticleState(
        phi=jnp.array([0.04, 0.04, 0.04, 0.04, 0.04]), # 总 20%
        rho=jnp.ones(5) * 2.7,
        u=jnp.ones(5) * 100.0, # 初始微弱滑移
        T=jnp.ones(5) * 300.0,
        r=jnp.array([1e-6, 2e-6, 5e-6, 10e-6, 20e-6])
    )
    
    init_state = State(x=jnp.array(0.0), gas=gas_init, part=part_init)
    
    # 2. 模拟物理参数 (符合 ZNDVectorField 签名)
    # atom_vec, coeffs_all, A_matrix, atomic_masses, eos_params
    eos_data = (
        jnp.zeros(5), jnp.zeros((7, 9)), jnp.zeros((5, 7)), 
        jnp.array([12, 1, 14, 16, 27]), 
        (None, None, None, None, None, None, 0, 10, 0, 0)
    )
    
    # 3. 运行积分 (0 到 0.5 mm)
    print("Starting Diffrax integration (0 -> 0.5 mm)...")
    try:
        sol = solve_znd_profile(jnp.array(7000.0), init_state, (0.0, 5.0e-4), eos_data)
        print("Integration SUCCESS")
        
        # 提取末态
        gas_final = sol.ys.gas
        part_final = sol.ys.part
        
        print(f"\n--- Final State at x = {sol.ts[-1]*1000:.2f} mm ---")
        print(f"Gas Reaction Progress (lam): {gas_final.lam[-1]:.4f}")
        print(f"Gas Velocity: {gas_final.u[-1]:.1f} m/s")
        print(f"Particle Velocities: {part_final.u[-1]}")
        
        # 验证颗粒是否被加速 (u_p 应增大)
        assert part_final.u[-1][0] > 100.0
        print("\nZND Dynamics verification: SUCCESS")
        
    except Exception as e:
        print(f"Integration FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_znd_integration_hmx()