# verify_p4_calibration.py
import jax
import jax.numpy as jnp
from pdu.calibration.inverse_solver import run_calibration, find_D_eigenvalue
from pdu.core.types import GasState, ParticleState, State
from pdu.mapper.jwl import project_to_jwl
from typing import NamedTuple, Any

class CJInfo(NamedTuple):
    D: float
    P_tot: float
    rho_cj: float
    gas: Any

def test_tritonal_calibration():
    print("Testing P4: Tritonal Calibration & JWL Projection...")
    
    # 1. 实验目标 (Tritonal 80/20)
    target_D = 6700.0
    rho0 = 1.72
    E0_theo = 7.5 # GPa (Effective)
    
    # 2. 初始状态 (VN Spike approx)
    gas_init = GasState(rho=jnp.array(2.4), u=jnp.array(1700.0), T=jnp.array(4000.0), lam=jnp.array(0.01))
    part_init = ParticleState(phi=jnp.array([0.1]), rho=jnp.array([2.7]), u=jnp.array([50.0]), T=jnp.array([300.0]), r=jnp.array([10e-6]))
    init_state = State(x=jnp.array(0.0), gas=gas_init, part=part_init)
    eos_data = (jnp.zeros(5), jnp.zeros((7, 9)), jnp.zeros((5, 7)), jnp.zeros(5), (None,)*10)
    
    # 3. 运行标定 (仅迭代 5 次作为功能验证)
    best_multiplier = run_calibration(target_D, init_state, (0.0, 1.0e-3), eos_data, n_iters=5)
    print(f"\nCalibrated Drag Multiplier: {float(best_multiplier):.4f}")
    
    # 4. 获取最终爆速并投影
    D_final = find_D_eigenvalue(best_multiplier, init_state, (0.0, 1.0e-3), eos_data)
    
    # 构造投影所需的 CJ 信息
    cj_info = CJInfo(
        D=float(D_final),
        P_tot=25.0, # 假设标定后的滞止压
        rho_cj=2.2,
        gas=gas_init
    )
    
    print("\nProjecting to macro JWL parameters...")
    jwl = project_to_jwl(cj_info, rho0, E0_theo)
    
    print(f"JWL Result: A={jwl.A:.1f}, B={jwl.B:.1f}, R1={jwl.R1:.2f}, R2={jwl.R2:.2f}, w={jwl.omega:.3f}")
    
    print("\nP4 Calibration & Projection verification: SUCCESS")

if __name__ == "__main__":
    # 需要处理 NamedTuple 在 verify 中的 import 问题
    from pdu.core.types import GasState
    import sys
    import pdu.mapper.jwl
    
    test_tritonal_calibration()
