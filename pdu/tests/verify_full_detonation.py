
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
from pdu.core.equilibrium import solve_equilibrium, build_stoichiometry_matrix
from pdu.data.products import load_products
from pdu.utils.precision import R_GAS, to_fp64
from pdu.physics.thermo import compute_enthalpy 
from pdu.physics.eos import JCZ3EOS, compute_pressure_jcz3, compute_internal_energy_jcz3 as compute_u_jcz3
from pdu.physics.eos import compute_total_helmholtz_energy
from pdu.physics.jwl import fit_jwl_from_isentrope, JWLParams

# Explosives Database (Same as before)
EXPLOSIVES = {
    "HMX": {
        "atoms": {'C': 4, 'H': 8, 'N': 8, 'O': 8}, 
        "rho": 1.90, "H_f": 75.0, "Mw": 296.16 # kJ/mol
    },
    "RDX": {
        "atoms": {'C': 3, 'H': 6, 'N': 6, 'O': 6},
        "rho": 1.80, "H_f": 70.0, "Mw": 222.12
    },
    "TNT": {
        "atoms": {'C': 7, 'H': 5, 'N': 3, 'O': 6},
        "rho": 1.64, "H_f": -63.0, "Mw": 227.13
    },
    "Tritonal": { # 0.8 TNT + 0.2 Al
        "atoms": {'C': 2.465, 'H': 1.761, 'N': 1.057, 'O': 2.113, 'Al': 0.741},
        "rho": 1.73, "H_f": -22.2, "Mw": 100.0 # Adjusted H_f
    },
    "PBXN-109": { # RDX 64% + Al 20% + HTPB 16% (CORRECTED)
        "atoms": {'C': 3.1, 'H': 5.6, 'N': 3.85, 'O': 3.85, 'Al': 1.48},
        "rho": 1.68, "H_f": -185.0, "Mw": 185.0  # Includes HTPB contribution
    }
}

SPECIES_LIST = ('N2', 'H2O', 'CO2', 'CO', 'C_graphite', 'H2', 'O2', 'OH', 'NO', 'Al2O3', 'Al', 'NH3', 'CH4')
ELEMENT_LIST = ('C', 'H', 'N', 'O', 'Al')

def get_thermo_data():
    products_db = load_products()
    coeffs_all = []
    for s in SPECIES_LIST:
        if s in products_db:
            c = products_db[s].coeffs_high
            if c.shape[0] > 7: c = c[:7]
            coeffs_all.append(c)
        else:
            coeffs_all.append(jnp.zeros(7))
    return jnp.stack(coeffs_all)

def get_thermo_and_eos(species_list, coeffs_all):
    # Load EOS Params
    eos = JCZ3EOS.from_species_list(species_list, coeffs_all)
    return eos

# Helper to calculate Internal Energy from Equilibrium (JCZ3 Version)
def calculate_state_E(atom_vec, V, T, A, coeffs_all, eos):
    # Pass eos params (tuple of matrices) to solve_equilibrium
    eos_params = (eos.epsilon_matrix, eos.r_star_matrix, eos.alpha_matrix)
    
    n_eq = solve_equilibrium(atom_vec, V, T, A, coeffs_all, eos_params)
    
    # E = U (Internal Energy)
    # Use JCZ3 Energy function
    E_total = compute_u_jcz3(n_eq, V, T, coeffs_all, eos.epsilon_matrix, eos.r_star_matrix, eos.alpha_matrix)
    
    # P (JCZ3 Pressure)
    P = compute_pressure_jcz3(n_eq, V, T, eos.epsilon_matrix, eos.r_star_matrix, eos.alpha_matrix)
    
    return E_total, P, n_eq

def find_cj_point(name, data, A, coeffs_all):
    # Initial State
    rho0 = data['rho'] # g/cm3
    Mw = data.get('Mw', 100.0)
    V0_m3_mol = (Mw / 1000.0) / (rho0 * 1000.0) # m3
    V0_cc = V0_m3_mol * 1e6 # cm3 (per mole/basis)
    
    E0_J = data['H_f'] * 1000.0 # kJ -> J (Formation Enthalpy)
    
    atom_vec = jnp.array([data['atoms'].get(e, 0.0) for e in ELEMENT_LIST])
    
    # Load EOS
    eos = get_thermo_and_eos(SPECIES_LIST, coeffs_all)
    
    # Simple Grid Search for CJ
    v_ratios = np.linspace(0.5, 0.85, 12) 
    
    results = []
    
    print(f"\nScanning {name} Hugoniot (JCZ3)...")
    for vr in v_ratios:
        V_test = V0_cc * vr
        
        # Solve Hugoniot for T
        T_min, T_max = 2000.0, 8000.0 
        for _ in range(12): 
            T_mid = (T_min + T_max) / 2
            E, P, _ = calculate_state_E(atom_vec, V_test, T_mid, A, coeffs_all, eos)
            
            dV_m3 = (V0_cc - V_test) * 1e-6
            Hugoniot_Term = 0.5 * P * dV_m3
            Err = (E - E0_J) - Hugoniot_Term
            
            if Err > 0: T_max = T_mid
            else: T_min = T_mid
        
        E_cj, P_cj, n_final = calculate_state_E(atom_vec, V_test, T_mid, A, coeffs_all, eos)
        
        rho0_kg = rho0 * 1000
        D_sq = P_cj / (rho0_kg * (1 - vr))
        if D_sq < 0: D_sq = 0
        D = np.sqrt(D_sq)
        
        n_sum = jnp.sum(n_final)
        results.append((D, P_cj/1e9, T_mid, vr))
    
    if not results: return (0,0,0,0)
    best = min(results, key=lambda x: x[0]) # CJ Point is minima on Hugoniot
    return best

def generate_isentrope_data(cj_V, cj_T, n_eq, atom_vec, A, coeffs_all, eos, num_points=15):
    """Generate Expansion Isentrope P-V data starting from CJ Point"""
    
    def compute_entropy_jcz3(n, V, T):
         eps, r_s, alpha = eos.epsilon_matrix, eos.r_star_matrix, eos.alpha_matrix
         grad_T_fn = jax.grad(compute_total_helmholtz_energy, argnums=2)
         dA_dT = grad_T_fn(n, V, T, coeffs_all, eps, r_s, alpha)
         return -dA_dT

    S_cj = compute_entropy_jcz3(n_eq, cj_V, cj_T)
    
    V_data = []
    P_data = []
    
    # Exp ratios > 1.0 (Expansion)
    exp_ratios = np.logspace(0, 0.9, num_points) # 1.0 to ~8.0
    
    current_T = cj_T
    
    for r in exp_ratios:
        V_curr = cj_V * r
        
        # Find T such that S(V, T) = S_cj
        T_low, T_high = 300.0, current_T * 1.5
        for _ in range(12):
             T_mid = (T_low + T_high) / 2
             
             eos_params = (eos.epsilon_matrix, eos.r_star_matrix, eos.alpha_matrix)
             n_new = solve_equilibrium(atom_vec, V_curr, T_mid, A, coeffs_all, eos_params)
             
             S_curr = compute_entropy_jcz3(n_new, V_curr, T_mid)
             
             if S_curr > S_cj: 
                 T_high = T_mid
             else:
                 T_low = T_mid
        
        T_iso = T_high
        current_T = T_iso
        
        eos_params = (eos.epsilon_matrix, eos.r_star_matrix, eos.alpha_matrix)
        n_iso = solve_equilibrium(atom_vec, V_curr, T_iso, A, coeffs_all, eos_params)
        P_iso = compute_pressure_jcz3(n_iso, V_curr, T_iso, *eos_params)
        
        V_data.append(V_curr)
        P_data.append(P_iso / 1e9) # GPa
        
    return V_data, P_data

def run():
    A = build_stoichiometry_matrix(SPECIES_LIST, ELEMENT_LIST)
    coeffs = get_thermo_data()
    
    print(f"{'Name':<15} | {'D (m/s)':<8} | {'P (GPa)':<8} | {'T (K)':<6} | {'JWL Parameters (A, B, R1, R2, w)'}")
    print("-" * 110)
    
    for name, data in EXPLOSIVES.items():
        try:
            # 1. Find CJ Point
            D, P_cj_gpa, T_cj, vr = find_cj_point(name, data, A, coeffs)
            
            # 2. Re-calculate CJ State for Isentrope
            rho0 = data['rho']
            Mw = data.get('Mw', 100.0)
            V0_cc = (Mw/1000.0)/(rho0*1000.0)*1e6
            V_cj = V0_cc * vr
            
            atom_vec = jnp.array([data['atoms'].get(e, 0.0) for e in ELEMENT_LIST])
            eos = get_thermo_and_eos(SPECIES_LIST, coeffs)
            
            eos_params = (eos.epsilon_matrix, eos.r_star_matrix, eos.alpha_matrix)
            n_cj = solve_equilibrium(atom_vec, V_cj, T_cj, A, coeffs, eos_params)
            
            # 3. Generate Isentrope
            V_iso, P_iso = generate_isentrope_data(V_cj, T_cj, n_cj, atom_vec, A, coeffs, eos)
            
            # 4. Fit JWL
            V_rel = np.array(V_iso) / V0_cc
            jwl = fit_jwl_from_isentrope(V_rel, np.array(P_iso), rho0, data['H_f'])
            
            jwl_str = f"A={jwl.A:.0f}, B={jwl.B:.1f}, R1={jwl.R1:.2f}, R2={jwl.R2:.2f}, w={jwl.omega:.2f}"
            
            print(f"{name:<15} | {D:.0f}     | {P_cj_gpa:.1f}     | {T_cj:.0f}   | {jwl_str}")
            
        except Exception as e:
            print(f"{name:<15} | Error: {e}")
            # import traceback
            # traceback.print_exc()

if __name__ == "__main__":
    run()
