
import jax
import jax.numpy as jnp
import json
import numpy as np
from pathlib import Path
from pdu.calibration.differentiable_cj_enhanced import predict_cj_with_isentrope
from pdu.core.equilibrium import build_stoichiometry_matrix
from pdu.data.products import load_products

def run_expanded_verification():
    # 1. Load Databases
    data_dir = Path('pdu/data')
    
    with open(data_dir / 'jcz3_params.json') as f:
        params_db = json.load(f)
        v7_params = params_db['species']
        
    with open(data_dir / 'reactants.json') as f:
        reactants_db = json.load(f)
        
    with open(data_dir / 'jwl_experimental.json') as f:
        jwl_db = json.load(f)

    # 2. Setup Physics Environment
    SPECIES_LIST = ('N2', 'H2O', 'CO2', 'CO', 'C_graphite', 'H2', 'O2', 'OH', 'NO', 'NH3', 'CH4', 'Al', 'Al2O3')
    ELEMENT_LIST = ('C', 'H', 'N', 'O', 'Al')
    atomic_masses = jnp.array([12.011, 1.008, 14.007, 15.999, 26.982])
    A_matrix = build_stoichiometry_matrix(SPECIES_LIST, ELEMENT_LIST)
    
    products_db = load_products()
    coeffs_all = []
    for s in SPECIES_LIST:
        coeffs_all.append(products_db[s].coeffs_high[:7] if s in products_db else jnp.zeros(7))
    coeffs_all = jnp.stack(coeffs_all)
    
    # 3. Prepare V7 Parameters
    reshaped = []
    for s in SPECIES_LIST:
        p = v7_params[s]
        reshaped.append([p['epsilon_over_k'], p['r_star'], p['alpha']])
    reshaped = jnp.array(reshaped)
    
    eps_vec = reshaped[:, 0]
    r_vec = reshaped[:, 1]
    alpha_vec = reshaped[:, 2]
    
    eps_matrix = jnp.sqrt(jnp.outer(eps_vec, eps_vec))
    r_matrix = (jnp.expand_dims(r_vec, 1) + jnp.expand_dims(r_vec, 0)) / 2.0
    alpha_matrix = jnp.sqrt(jnp.outer(alpha_vec, alpha_vec))

    # 4. Map Reactants for Verification
    # Some names might differ slightly between databases
    name_map = {
        "HMX": ("explosives", "HMX"),
        "RDX": ("explosives", "RDX"),
        "TNT": ("explosives", "TNT"),
        "PETN": ("explosives", "PETN"),
        "NM": ("explosives", "NM"),
        "Comp_B": ("mixtures", "Comp_B"),
        "Octol_75_25": ("mixtures", "Octol_75_25"),
        "Tritonal": ("mixtures", "Tritonal"),
        "PBXN_109": ("mixtures", "PBXN_109")
    }

    results = []

    print("="*100)
    print(f"{'Explosive':<15} | {'Type':<10} | {'D_exp':<8} | {'D_v7':<8} | {'D_err':<8} | {'P_exp':<8} | {'P_v7':<8} | {'P_err':<8}")
    print("-" * 100)

    for exp_name, (cat, key) in name_map.items():
        if key not in reactants_db[cat]:
            print(f"Warning: {key} not found in reactants database.")
            continue
            
        reactant = reactants_db[cat][key]
        target = jwl_db['explosives'].get(exp_name)
        
        if not target:
            print(f"Warning: No experimental targets for {exp_name}.")
            continue

        # Get Atom Vector and Props
        if "effective_formula" in reactant:
            atoms_dict = reactant["effective_formula"]
            Hf = reactant["effective_Hf"]
            Mw = reactant["effective_Mw"]
        else:
            atoms_dict = reactant["formula"]
            Hf = reactant["enthalpy_formation"]
            Mw = reactant["molecular_weight"]
            
        av = [atoms_dict.get(e, 0) for e in ELEMENT_LIST]
        atom_vec = jnp.array(av)
        
        rho = target['rho0']
        V0_calc = (Mw / 1000.0) / (rho * 1000.0) * 1e6
        E0_calc = Hf * 1000.0

        try:
            # Predict
            D_v7, P_v7, T_v7, V_v7, _, _ = predict_cj_with_isentrope(
                eps_matrix, r_matrix, alpha_matrix,
                V0_calc, E0_calc, atom_vec, coeffs_all, A_matrix, atomic_masses, 5
            )
            
            D_v7 = float(D_v7)
            P_v7 = float(P_v7)
            
            D_err = (D_v7 - target['D_exp']) / target['D_exp'] * 100
            P_err = (P_v7 - target['P_CJ_exp']) / target['P_CJ_exp'] * 100
            
            results.append({
                "name": exp_name,
                "type": cat,
                "D_exp": target['D_exp'],
                "D_v7": D_v7,
                "D_err": D_err,
                "P_exp": target['P_CJ_exp'],
                "P_v7": P_v7,
                "P_err": P_err
            })
            
            print(f"{exp_name:<15} | {cat:<10} | {target['D_exp']:<8.0f} | {D_v7:<8.0f} | {D_err:<+7.1f}% | {target['P_CJ_exp']:<8.1f} | {P_v7:<8.1f} | {P_err:<+7.1f}%")

        except Exception as e:
            print(f"{exp_name:<15} | Error: {e}")

    print("="*100)
    
    # Summary Table
    print("\nSUMMARY OF ERRORS:")
    avg_d_err = np.mean([abs(r['D_err']) for r in results])
    avg_p_err = np.mean([abs(r['P_err']) for r in results])
    print(f"Average Absolute Error D: {avg_d_err:.2f}%")
    print(f"Average Absolute Error P: {avg_p_err:.2f}%")
    
    # Save results to JSON for reporting
    with open('pdu/tests/verification_v7_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_expanded_verification()
