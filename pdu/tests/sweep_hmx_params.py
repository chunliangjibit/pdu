# tests/sweep_hmx_params.py
import jax
import jax.numpy as jnp
from pdu.solver.jump_conditions import compute_vn_spike
from pdu.solver.znd import solve_znd_projection
from pdu.data.products import load_products
from pdu.core.equilibrium import build_stoichiometry_matrix
from pdu.thermo.implicit_eos import get_thermo_properties
import json

def run_hmx_point(r_scale, alpha_val, q_reaction_val=5.8e6):
    # Data from calibrate_hmx_v11.py
    rho0 = 1.891
    D = 9110.0
    e0 = 75000.0 / 0.29616
    q_reaction = q_reaction_val
    
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
        # Use database r_star but scaled
        base_r = p.get('r_star', 3.5)
        # Cap N2/CO2 to avoid excessive overlap
        if s in ('N2', 'CO2'): base_r = min(base_r, 4.0)
        r.append(base_r * r_scale) 
        alpha.append(alpha_val)
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

    # Use Fixed baseline logic (e0 - q)
    try:
        # We target CJ state. Since we don't have a full CJ solver here, 
        # we estimate CJ properties using ZND find-sonic-point logic or just a simplified jump
        # For sweep speed, we just find the point on Hugoniot that reaches lam=1
        # Actually, let's just use solve_znd and check final state
        
        rho_vn, u_vn, T_vn = compute_vn_spike(D, rho0, e0 - q_reaction, 1e5, *eos_data)
        
        from pdu.core.types import GasState, ParticleState, State
        gas_init = GasState(rho=rho_vn, u=u_vn, T=T_vn, lam=jnp.array(1e-4))
        init_state = State(gas=gas_init, part=None)
        
        sol = solve_znd_projection(D, init_state, (0.0, 1e-4), eos_data, None, None, q_reaction)
        
        P_f, _ = get_thermo_properties(sol.rho[-1], sol.T[-1], *eos_data)
        return float(P_f/1e9), float(sol.T[-1])
    except:
        return None, None

def sweep():
    print("r_scale | alpha | P_CJ (GPa) | T_CJ (K)")
    print("--------|-------|------------|---------")
    for r_sc in [0.8, 0.85, 0.9, 0.95]:
        for alp in [13.0, 13.5]:
            p, t = run_hmx_point(r_sc, alp)
            if p:
                print(f"{r_sc:7.2f} | {alp:5.1f} | {p:10.2f} | {t:7.0f}")
            else:
                print(f"{r_sc:7.2f} | {alp:5.1f} |   FAILED   |  -")

if __name__ == "__main__":
    sweep()
