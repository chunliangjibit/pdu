# pdu/tests/calibrate_hmx_v11.py
import jax
import jax.numpy as jnp
import optax
from pdu.core.types import GasState, ParticleState, State
from pdu.solver.znd import solve_znd_profile
from pdu.solver.jump_conditions import compute_vn_spike
from pdu.thermo.implicit_eos import get_thermo_properties, get_sound_speed
from pdu.flux import IgraDrag, RanzMarshallHeat
from pdu.data.components import get_component
from pdu.data.products import load_products
from pdu.core.equilibrium import build_stoichiometry_matrix
import json
from pathlib import Path

def calibrate_hmx():
    print("=== HMX Calibration (V11.0 Differentiable Core) ===")
    
    # 1. HMX 实验数据
    rho0 = 1.891
    D_exp = 9110.0
    P_exp = 39.0
    
    # 2. 物理环境加载
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
        r.append(3.1) # FORCE 3.1 (Override JSON to fix 992 GPa overpacking)
        alpha.append(p.get('alpha', 13.0))
        lam.append(p.get('lambda_ree', 0.0))
    
    # V11 Phase 4: Correct Solid Mask for C_graphite (idx 4)
    solid_mask = jnp.array([0., 0., 0., 0., 1., 0., 0.])
    solid_v0_vec = jnp.array([0., 0., 0., 0., 5.3, 0., 0.]) # cm3/mol
    
    eos_params = (jnp.array(eps), jnp.array(r), jnp.array(alpha), jnp.array(lam),
                  solid_mask, solid_v0_vec, 
                  0.0, 10.0, 0.0, 0.0) # r_star_rho_corr = 0.0
    
    products_db = load_products()
    coeff_list_low = []
    coeff_list_high = []
    for s in SPECIES_LIST:
        p = products_db[s]
        # Pad NASA-7 to length 9
        cl = p.coeffs_low[:7]
        ch = p.coeffs_high[:7]
        coeff_list_low.append(jnp.concatenate([jnp.zeros(2), cl]))
        coeff_list_high.append(jnp.concatenate([jnp.zeros(2), ch]))
    
    coeffs_low = jnp.stack(coeff_list_low)
    coeffs_high = jnp.stack(coeff_list_high)
    
    # HMX: C4H8N8O8
    atom_vec = jnp.array([4.0, 8.0, 8.0, 8.0, 0.0])
    eos_data = (atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params)
    
    # 理论反应热与初始能量 (J/kg)
    # HMX 形成焓 Hf = 75 kJ/mol. Mw = 0.29616 kg/mol.
    e0 = 75000.0 / 0.29616
    q_reaction = 5.8e6 
    
    # 3. 初始猜测 D
    D_var = jnp.array(D_exp) 
    
    # 4. 计算 VN Spike
    rho_vn, u_vn, T_vn = compute_vn_spike(D_var, rho0, e0, 1e5, *eos_data)
    
    # 检查 VN 点声速
    p_vn, n_vn = get_thermo_properties(rho_vn, T_vn, *eos_data)
    # Correct unpack: atom_vec, cl, ch, A, masses, params = eos_data
    a_vn = get_sound_speed(rho_vn, T_vn, n_vn, eos_data[0], eos_data[1], eos_data[2], eos_data[5], eos_data[4])
    M_vn = u_vn / a_vn
    print(f"VN Spike: rho={rho_vn:.3f}, u={u_vn:.1f}, T={T_vn:.0f} K, a={a_vn:.1f}, M={M_vn:.3f}")
    
    # 5. 运行正向剖面
    gas_init = GasState(rho=rho_vn, u=u_vn, T=T_vn, lam=jnp.array(1e-4))
    part_init = ParticleState(phi=jnp.array([1e-10]), rho=jnp.array([2.7]), u=jnp.array([D_var]), T=jnp.array([300.0]), r=jnp.array([1e-5]))
    init_state = State(gas=gas_init, part=part_init)
    
    drag_m = IgraDrag(init_val=1.0)
    heat_m = RanzMarshallHeat(init_val=1.0)
    
    print("\nRunning ZND Profile for HMX...")
    sol = solve_znd_profile(D_var, init_state, (0.0, 2e-4), eos_data, drag_m, heat_m, q_reaction)
    
    # 检查末态
    # 检查末态
    rho_f = sol.gas.rho[-1]
    T_f = sol.gas.T[-1]
    u_f = sol.gas.u[-1]
    
    P_f, _ = get_thermo_properties(rho_f, T_f, *eos_data)
    
    print(f"Final State (lam={sol.gas.lam[-1]:.3f}, Steps={len(sol.x)}):")
    print(f"  P = {P_f/1e9:.2f} GPa (Exp: {P_exp:.2f})")
    print(f"  u = {u_f:.1f} m/s")
    
    # 打印剖面简报
    print("\nProfile Overview (lam start):")
    print(sol.gas.lam[:10])
    
    print("\nTrajectory Tail (Last 10 points):")
    print(f"P (GPa): {sol.P[-10:]/1e9}")
    print(f"T (K):   {sol.T[-10:]}")
    print(f"u (m/s): {sol.u[-10:]}")
    print(f"lam:     {sol.gas.lam[-10:]}")
    
    # 计算误差
    p_err = (P_f/1e9 - P_exp) / P_exp
    print(f"Pressure Error: {p_err*100:.1f}%")

if __name__ == "__main__":
    calibrate_hmx()
