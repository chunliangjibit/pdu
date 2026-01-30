# pdu/calibration/inverse_solver.py
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Tuple, Any

from pdu.core.types import GasState, ParticleState, State
from pdu.flux import IgraDrag, RanzMarshallHeat
from pdu.solver.znd import shooting_residual, solve_znd_profile

"""
PDU V11.0 P4: 逆向标定算子 (IFT 增强版)
使用隐函数定理 (IFT) 实现对特征爆速 D 的精确微分。
"""

@jax.custom_vjp
def find_D_eigenvalue(drag_multiplier, init_state, x_span, eos_data):
    """
    寻找特征爆速 D。前向使用简单的二分法搜索。
    """
    def residual_fn(d):
        drag_m = IgraDrag(init_val=drag_multiplier)
        heat_m = RanzMarshallHeat(init_val=1.0)
        return shooting_residual(d, init_state, x_span, eos_data, drag_m, heat_m)

    # 前向二分法 (非可微循环)
    def bisection(lo, hi):
        def step(i, bounds):
            l, h = bounds
            m = (l + h) / 2.0
            return jnp.where(residual_fn(m) > 0, l, m), jnp.where(residual_fn(m) > 0, m, h)
        final_bounds = jax.lax.fori_loop(0, 15, step, (lo, hi))
        return (final_bounds[0] + final_bounds[1]) / 2.0

    return bisection(6000.0, 9000.0)

def find_D_fwd(drag_multiplier, init_state, x_span, eos_data):
    D_sol = find_D_eigenvalue(drag_multiplier, init_state, x_span, eos_data)
    return D_sol, (D_sol, drag_multiplier, init_state, x_span, eos_data)

def find_D_bwd(res, g):
    """
    IFT 伴随梯度： dD/dtheta = - (dR/dtheta) / (dR/dD)
    """
    D_sol, multiplier, init_state, x_span, eos_data = res
    
    def residual_fixed_theta(d):
        drag_m = IgraDrag(init_val=multiplier)
        heat_m = RanzMarshallHeat(init_val=1.0)
        return shooting_residual(d, init_state, x_span, eos_data, drag_m, heat_m)
    
    def residual_fixed_D(m):
        drag_m = IgraDrag(init_val=m)
        heat_m = RanzMarshallHeat(init_val=1.0)
        return shooting_residual(D_sol, init_state, x_span, eos_data, drag_m, heat_m)

    # 计算偏导数
    dR_dD = jax.grad(residual_fixed_theta)(D_sol)
    dR_dtheta = jax.grad(residual_fixed_D)(multiplier)
    
    # 避免除以 0
    dD_dtheta = - dR_dtheta / (dR_dD + 1e-10)
    
    # 返回对 drag_multiplier 的梯度 (g 是上游传递的梯度)
    return g * dD_dtheta, None, None, None

find_D_eigenvalue.defvjp(find_D_fwd, find_D_bwd)

def calibration_loss(drag_multiplier, target_D, init_state, x_span, eos_data):
    D_sim = find_D_eigenvalue(drag_multiplier, init_state, x_span, eos_data)
    return jnp.square(D_sim - target_D)

def run_calibration(target_D_exp, init_state, x_span, eos_data, n_iters=10):
    multiplier = jnp.array(1.0)
    optimizer = optax.adam(learning_rate=0.05)
    opt_state = optimizer.init(multiplier)
    
    @jax.jit
    def step(m, s):
        loss, grads = jax.value_and_grad(calibration_loss)(m, target_D_exp, init_state, x_span, eos_data)
        updates, next_s = optimizer.update(grads, s)
        next_m = optax.apply_updates(m, updates)
        return next_m, next_s, loss

    print(f"Starting Differentiable Calibration to target D = {target_D_exp:.1f} m/s...")
    
    current_m = multiplier
    for i in range(n_iters):
        current_m, opt_state, l_val = step(current_m, opt_state)
        print(f"  Iteration {i}: Multiplier = {float(current_m):.44f}, Loss = {float(l_val):.2e}")
        
    return current_m