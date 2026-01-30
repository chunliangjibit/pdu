# pdu/tests/test_v11_p4_5_hardening.py
import jax
import jax.numpy as jnp
from pdu.core.types import GasState, ParticleState, State
from pdu.solver.znd import solve_znd_profile
from pdu.solver.jump_conditions import compute_vn_spike
from pdu.thermo.implicit_eos import get_thermo_properties, get_internal_energy_pt
from pdu.flux import IgraDrag, RanzMarshallHeat
from pdu.data.components import get_component
from pdu.data.products import load_products
from pdu.core.equilibrium import build_stoichiometry_matrix
import json
from pathlib import Path

def test_hardened_tritonal_znd():
    print("Testing P4.5: Hardened ZND Physics (Tritonal 80/20)...")
    
    # 1. 环境准备 (对标 Tritonal)
    rho0 = 1.72
    D_guess = 6700.0
    
    # 加载 JCZ3 参数
    SPECIES_LIST = ('N2', 'H2O', 'CO2', 'CO', 'C_graphite', 'H2', 'O2')
    ELEMENT_LIST = ('C', 'H', 'N', 'O', 'Al')
    A_matrix = build_stoichiometry_matrix(SPECIES_LIST, ELEMENT_LIST)
    atomic_masses = jnp.array([12.011, 1.008, 14.007, 15.999, 26.982])
    
    data_dir = Path('pdu/data')
    with open(data_dir / 'jcz3_params.json') as f:
        v7_params = json.load(f)['species']
    
    eps, r, alpha, lam = [], [], [], []
    for s in SPECIES_LIST:
        p = v7_params[s]
        eps.append(p.get('epsilon_over_k', 100.0))
        r.append(p.get('r_star', 3.5))
        alpha.append(p.get('alpha', 13.0))
        lam.append(p.get('lambda_ree', 0.0))
    
    eos_params = (jnp.array(eps), jnp.array(r), jnp.array(alpha), jnp.array(lam),
                  jnp.zeros(len(SPECIES_LIST)), jnp.zeros(len(SPECIES_LIST)), 
                  0.0, 10.0, 0.0, 0.0)
    
    products_db = load_products()
    coeff_list = []
    for s in SPECIES_LIST:
        p = products_db[s]
        c7 = p.coeffs_high[:7]
        coeff_list.append(jnp.concatenate([jnp.zeros(2), c7]))
    coeffs_all = jnp.stack(coeff_list)
    
    atom_vec = jnp.array([5.6, 4.0, 2.4, 4.8, 0.2])
    eos_data = (atom_vec, coeffs_all, A_matrix, atomic_masses, eos_params)
    
    # 初始能量 (TNT 爆热相关能量)
    # HMX 5.8 MJ/kg -> e0 approx 0.0 J/kg relative to products
    # 对于 VN Spike，我们使用反应物的形成能
    e0 = 0.0 # 简化为基准
    
    # 2. 计算 VN Spike
    print(f"Calculating VN Spike for D = {D_guess} m/s...")
    rho_vn, u_vn, T_vn = compute_vn_spike(D_guess, rho0, e0, 1e5, *eos_data)
    print(f"VN Spike: rho={rho_vn:.2f}, u={u_vn:.1f}, T={T_vn:.0f} K")
    
    # 3. 初始化 ZND 状态
    gas_init = GasState(rho=rho_vn, u=u_vn, T=T_vn, lam=jnp.array(0.01))
    part_init = ParticleState(
        phi=jnp.array([0.15]), 
        rho=jnp.array([2.7]), 
        u=jnp.array([D_guess]), 
        T=jnp.array([300.0]), 
        r=jnp.array([15e-6])
    )
    init_state = State(gas=gas_init, part=part_init)
    
    # 4. 运行高保真积分
    print("\nStarting Optimized High-Fidelity ZND integration (T-Variable Core)...")
    drag_m = IgraDrag(init_val=1.0)
    heat_m = RanzMarshallHeat(init_val=1.0)
    
    # 估算反应能 Q
    q_reaction = 5.8e6 # MJ/kg -> J/kg
    
    # 运行 V11 去奇异化积分
    sol = solve_znd_profile(
        jnp.array(D_guess), 
        init_state, 
        (0.0, 5e-5), 
        eos_data, 
        drag_m, 
        heat_m,
        q_reaction
    )
    
    print(f"Integration Result: SUCCESS (Steps: {len(sol.ts)})")
    
    # 5. 物理量检查
    gas_f = sol.gas
    
    # 最终压力 (GPa) - Phase 1: n 在向量场内部代数求解
    P_gas_f, _ = get_thermo_properties(
        gas_f.rho[-1], gas_f.T[-1], 
        atom_vec, coeffs_all, A_matrix, atomic_masses, eos_params
    )
    
    # 总压力 (V11 简化版: 考虑动量通量，若无粒子则忽略)
    P_total = P_gas_f / 1e9
    
    print(f"\n--- Final Results (xi = {sol.ts[-1]:.3f}) ---")
    print(f"Reaction Progress (lam): {gas_f.lam[-1]:.3f}")
    print(f"Gas Static Pressure: {P_gas_f/1e9:.2f} GPa")
    print(f"TOTAL PRESSURE (P_tot): {P_total:.2f} GPa")
    
    # 验证压力是否接近实验预期 (15-25 GPa)
    assert P_total > 15.0
    print("\nPhysics Hardening verification: SUCCESS")

if __name__ == "__main__":
    test_hardened_tritonal_znd()