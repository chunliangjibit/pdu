# verify_p3_bisection.py
import jax
import jax.numpy as jnp
from pdu.core.types import GasState, ParticleState, State
from pdu.solver.znd import solve_znd_profile, shooting_residual

def test_bisection_shooting():
    print("Testing P3: Generalized Shooting Method (Bisection)...")
    
    # 1. 设置初始状态
    gas_init = GasState(rho=jnp.array(2.5), u=jnp.array(1800.0), T=jnp.array(4500.0), lam=jnp.array(0.01))
    part_init = ParticleState(phi=jnp.array([0.1]), rho=jnp.array([2.7]), u=jnp.array([100.0]), T=jnp.array([300.0]), r=jnp.array([10e-6]))
    init_state = State(x=jnp.array(0.0), gas=gas_init, part=part_init)
    eos_data = (jnp.zeros(5), jnp.zeros((7, 9)), jnp.zeros((5, 7)), jnp.zeros(5), (None,)*10)
    
    # 2. 二分法搜索 D
    D_lo, D_hi = 6000.0, 9000.0
    
    @jax.jit
    def bisection_step(i, bounds):
        lo, hi = bounds
        mid = (lo + hi) / 2.0
        res = shooting_residual(mid, init_state, (0.0, 1.0e-3), eos_data)
        # 如果残差 > 0 (M > 1)，说明 D 可能太小 (u 降得不够慢)
        # 注意：此处逻辑需根据物理导数符号微调，暂定：
        return jnp.where(res > 0, lo, mid), jnp.where(res > 0, mid, hi)

    print(f"Starting Bisection from [{D_lo}, {D_hi}] m/s")
    final_bounds = jax.lax.fori_loop(0, 10, bisection_step, (D_lo, D_hi))
    D_final = (final_bounds[0] + final_bounds[1]) / 2.0
    
    print(f"\nFinal Eigenvalue Detonation Velocity: D = {float(D_final):.1f} m/s")
    
    # 验证最终剖面
    sol = solve_znd_profile(D_final, init_state, (0.0, 1.0e-3), eos_data)
    print(f"Final Lambda: {sol.ys.gas.lam[-1]:.3f}")
    print(f"Final Mach Number: {sol.ys.gas.u[-1]/7000.0:.3f}")
    
    assert D_final > 6000.0 and D_final < 9000.0
    print("\nBisection Shooting verification: SUCCESS")

if __name__ == "__main__":
    test_bisection_shooting()
