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

# V11 Phase 6: JWL Fitting Imports
from pdu.thermo.isentrope import generate_isentrope
from pdu.physics.jwl import fit_jwl_from_isentrope, jwl_pressure
from pdu.utils.precision import to_fp64

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
        base_r = p.get('r_star', 3.5)
        # V11 Refinement: Use moderate scaling around 0.84 to balance repulsion and T
        r.append(base_r * 0.845) 
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
    q_reaction = 5.8e6 * 0.76 # Effective detonation heat (Gurney consistent)
    
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
    
    # =========================================================================
    # V11 Phase 6: JWL Fitting (Automated)
    # =========================================================================
    if abs(p_err) < 0.15: # Only fit if result is reasonable
        print("\n=== Generating Isentrope for JWL Fitting ===")
        # 1. Generate Isentrope Data (V_cj to 10*V_cj)
        # We need V relative to V0 (rho0=1.891) for JWL standard.
        # V0 = 1/rho0. 
        # V_cj = 1/rho_f.
        # V_rel_start = V_cj / (1.0/rho0) = rho0 / rho_f
        
        # Calculate Isentrope using new solver
        # Note: generate_isentrope returns arrays of P(Pa), T(K) for RELATIVE volumes V/V_cj
        # wait, generate_isentrope implementation assumes V_array input is absolute?
        # Let's check isentrope.py signature again.
        # It takes V_max_rel scalar. It generates V_array internally.
        # And returns V, P, T.
        
        # Re-check generate_isentrope input:
        # V_array = V_cj * V_factors (where factors 1.0 -> v_max_rel)
        # So return V_array is absolute volume (m3 for the recipe unit).
        
        V_abs, P_isen, T_isen = generate_isentrope(
             rho_f, T_f, atom_vec, coeffs_low, coeffs_high, A_matrix, atomic_masses, eos_params,
             v_max_rel=10.0, n_points=25
        )
        
        # Convert to JWL relative volume: V_jwl = v / v0
        # v0 (specific volume) = 1/rho0
        # v (specific volume) = V_abs / m_kg
        
        # Alternatively: V_jwl = V_abs / V_abs_0
        # where V_abs_0 = m_kg / rho0
        # This is ratio-consistent.
        
        am_kg = jnp.where(jnp.max(atomic_masses)>0.5, atomic_masses*1e-3, atomic_masses)
        m_kg = jnp.dot(atom_vec, am_kg)
        V_abs_0 = m_kg / (rho0 * 1000.0) # m3
        
        V_rel_jwl = V_abs / V_abs_0
        P_gpa = P_isen / 1e9
        
        print(f"Isentrope Generated: {len(P_gpa)} points. Range V_rel [{V_rel_jwl[0]:.2f}, {V_rel_jwl[-1]:.2f}]")
        print(f"P range: {P_gpa[0]:.2f} -> {P_gpa[-1]:.4f} GPa")
        
        # 2. Fit JWL
        print("\n=== Fitting JWL Parameters ===")
        # E0 in GPa (Energy per unit volume). 
        # e0 is J/kg. rho0 is g/cm3 -> 1891 kg/m3.
        # E0 = e0 * rho0_si = (J/kg) * (kg/m3) = J/m3 = Pa. /1e9 -> GPa.
        
        # e0 calculated above: 75000 / 0.296 = 2.5e5 J/kg? No.
        # HMX Hf=75 kJ/mol. Mw=0.296 kg/mol.
        # e0 = 75000 / 0.296 = 253378 J/kg.
        # Wait, heat of detonation is ~5-6 MJ/kg. e0 (internal energy of formation) is smaller.
        # But JWL E0 is "Total Energy available for expansion work".
        # It is usually (e_init - e_products_at_ref)?
        # Or often user supplied?
        # Let's use the provided q_reaction for estimating E0 initially, or e0 * rho0.
        # The fitter manual says "E0 (GPa) initial energy density".
        # Typically for HMX E0 ~ 9-10 GPa.
        # q_reaction = 5.8e6 J/kg. 
        # E0 ~ 5.8e6 * 1891 ~ 1.1e10 Pa = 11 GPa.
        
        E0_gpa = (q_reaction * rho0 * 1000.0) / 1e9
        
        jwl = fit_jwl_from_isentrope(
            V_rel_jwl, P_gpa, rho0, E0_gpa, D_exp, P_f/1e9,
            method='RELAXED_PENALTY'
        )
        
        print("\n=== Final JWL Calibration Results ===")
        print(f"A = {jwl.A:.4f}")
        print(f"B = {jwl.B:.4f}")
        print(f"R1 = {jwl.R1:.4f}")
        print(f"R2 = {jwl.R2:.4f}")
        print(f"omega = {jwl.omega:.4f}")
        print(f"E0 = {jwl.E0:.4f}")
        print(f"MSE (Log) = {jwl.fit_mse:.6e}")
        
    else:
        print("\nSkipping JWL Fitting: P_CJ error too high.")

if __name__ == "__main__":
    calibrate_hmx()
