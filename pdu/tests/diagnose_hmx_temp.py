# tests/diagnose_hmx_temp.py
import jax
import jax.numpy as jnp
from pdu.solver.jump_conditions import compute_vn_spike
from pdu.solver.znd import solve_znd_projection
from pdu.data.products import load_products
from pdu.core.equilibrium import build_stoichiometry_matrix
import json

def test_hmx_baseline():
    print("=== HMX Temperature Anomaly Diagnosis ===")
    
    # Data from calibrate_hmx_v11.py
    rho0 = 1.891
    D = 9110.0
    e0 = 75000.0 / 0.29616
    q_reaction = 5.8e6
    
    SPECIES_LIST = ('N2', 'H2O', 'CO2', 'CO', 'C_graphite', 'H2', 'O2')
    ELEMENT_LIST = ('C', 'H', 'N', 'O', 'Al')
    A_matrix = build_stoichiometry_matrix(SPECIES_LIST, ELEMENT_LIST)
    atomic_masses = jnp.array([12.011, 1.008, 14.007, 15.999, 26.982])
    
    with open('pdu/data/jcz3_params.json') as f:
        v7 = json.load(f)['species']
    
    eps, r, alpha, lam = [], [], [], []
    for s in SPECIES_LIST:
        p = v7[s]
        eps.append(p.get('epsilon_over_k', 100.0))
        r.append(3.045) 
        alpha.append(p.get('alpha', 13.0))
        lam.append(p.get('lambda_ree', 0.0))
    
    solid_mask = jnp.array([0., 0., 0., 0., 1., 0., 0.])
    solid_v0_vec = jnp.array([0., 0., 0., 0., 5.3, 0., 0.])
    eos_params = (jnp.array(eps), jnp.array(r), jnp.array(alpha), jnp.array(lam),
                  solid_mask, solid_v0_vec, 
                  0.0, 10.0, 0.0, 0.0)
    
    products_db = load_products()
    cl_l, ch_l = [], []
    for s in SPECIES_LIST:
        p = products_db[s]
        cl_l.append(jnp.concatenate([jnp.zeros(2), p.coeffs_low[:7]]))
        ch_l.append(jnp.concatenate([jnp.zeros(2), p.coeffs_high[:7]]))
    
    atom_vec = jnp.array([4.0, 8.0, 8.0, 8.0, 0.0])
    eos_data = (atom_vec, jnp.stack(cl_l), jnp.stack(ch_l), A_matrix, atomic_masses, eos_params)

    # Case A: Current (Buggy) - Pass e0 to VN spike
    print("\n--- Case A: Current Implementation (e0 passed to VN) ---")
    rho_vn_A, u_vn_A, T_vn_A = compute_vn_spike(D, rho0, e0, 1e5, *eos_data)
    print(f"VN Spike A: T = {T_vn_A:.0f} K")
    
    # Case B: Fixed - Pass (e0 - q_reaction) to VN spike
    print("\n--- Case B: Fixed Implementation (e0 - q_reaction passed to VN) ---")
    # This reflects that the UNREACTED state has q_reaction potential energy 
    # but its THERMAL state matches the products at a lower baseline.
    rho_vn_B, u_vn_B, T_vn_B = compute_vn_spike(D, rho0, e0 - q_reaction, 1e5, *eos_data)
    print(f"VN Spike B: T = {T_vn_B:.0f} K")

if __name__ == "__main__":
    test_hmx_baseline()
