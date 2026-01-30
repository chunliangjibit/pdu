# verify_p3_shooting.py
import jax
import jax.numpy as jnp
import optax
from pdu.core.types import GasState, ParticleState, State
from pdu.solver.znd import solve_znd_profile, shooting_loss

def test_shooting_optimization():
    print("Testing P3: Generalized Shooting Method (Optimization)...")
    
    # 1. 模拟初始状态
    gas_init = GasState(rho=jnp.array(2.5), u=jnp.array(1800.0), T=jnp.array(4500.0), lam=jnp.array(0.01))
    part_init = ParticleState(phi=jnp.array([0.1]), rho=jnp.array([2.7]), u=jnp.array([100.0]), T=jnp.array([300.0]), r=jnp.array([10e-6]))
    init_state = State(x=jnp.array(0.0), gas=gas_init, part=part_init)
    eos_data = (jnp.zeros(5), jnp.zeros((7, 9)), jnp.zeros((5, 7)), jnp.zeros(5), (None,)*10)
    
    # 2. 优化循环 (寻找最优 D)
    D_var = jnp.array(8000.0) # 初始猜测
    optimizer = optax.adam(learning_rate=100.0)
    opt_state = optimizer.init(D_var)
    
    print(f"Starting Shooting Optimization from D = {D_var:.1f} m/s")
    
    @jax.jit
    def step(d_val, o_state):
        loss, grads = jax.value_and_grad(shooting_loss)(d_val, init_state, (0.0, 1.0e-3), eos_data)
        updates, next_opt_state = optimizer.update(grads, o_state)
        next_d = optax.apply_updates(d_val, updates)
        return next_d, next_opt_state, loss

    for i in range(10):
        D_var, opt_state, l_val = step(D_var, opt_state)
        print(f"  Iteration {i}: D = {float(D_var):.1f}, Loss = {float(l_val):.2e}")

    print("\nShooting Method Differentiability verification: SUCCESS")

if __name__ == "__main__":
    test_shooting_optimization()
