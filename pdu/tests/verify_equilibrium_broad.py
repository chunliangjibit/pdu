
import jax
import jax.numpy as jnp
import numpy as np
from pdu.core.equilibrium import solve_equilibrium, build_stoichiometry_matrix
from pdu.data.products import load_products
from pdu.utils.precision import R_GAS

# 1. Database of Explosives (Simplified for testing)
# Format: Name -> {Elements: Moles}, Density (g/cm3), Approx CJ State (V cc/g, T K)
# Note: V_cj is roughly 1/rho * (gamma/(gamma+1)). Gamma ~ 3 -> V_cj ~ 0.75 * V0
EXPLOSIVES = {
    "HMX": {
        "atoms": {'C': 4, 'H': 8, 'N': 8, 'O': 8}, 
        "rho": 1.90, "V_est": 0.4, "T_est": 3800
    },
    "RDX": {
        "atoms": {'C': 3, 'H': 6, 'N': 6, 'O': 6},
        "rho": 1.80, "V_est": 0.42, "T_est": 3800
    },
    "TNT": {
        "atoms": {'C': 7, 'H': 5, 'N': 3, 'O': 6},
        "rho": 1.64, "V_est": 0.46, "T_est": 3000
    },
    "Octol (75/25)": { # 0.75 HMX + 0.25 TNT (approx mole mix) -> C=4.75, H=7.25, N=6.75, O=7.5
        # Mass ratio to Mole ratio: HMX(296), TNT(227). 75g/296=0.253 mol, 25g/227=0.110 mol.
        # Norm to 1 mol total approx? Or just sum atoms.
        # Let's use 1kg basis or arbitrary. Summing atoms is fine for Equilibrium.
        "atoms": {'C': 4.9, 'H': 7.1, 'N': 6.5, 'O': 7.4}, # Rough approx
        "rho": 1.81, "V_est": 0.41, "T_est": 3600
    },
    "Tritonal (80/20)": { # 80% TNT + 20% Al
        # TNT(227), Al(27). 80g TNT=0.352mol, 20g Al=0.741mol.
        # Atoms: 0.352 * C7H5N3O6 + 0.741 * Al
        "atoms": {'C': 2.46, 'H': 1.76, 'N': 1.05, 'O': 2.11, 'Al': 0.74},
        "rho": 1.73, "V_est": 0.43, "T_est": 4200 # Higher T due to Al
    },
    "PBXN-109": { # RDX/Al/Binder ~ 64/20/16. 
        # Complex. Let's approximate RDX + Al. 
        "atoms": {'C': 1.9, 'H': 3.8, 'N': 3.8, 'O': 3.8, 'Al': 1.5}, # Dummy high Al
        "rho": 1.65, "V_est": 0.45, "T_est": 4500
    }
}

SPECIES_LIST = ('N2', 'H2O', 'CO2', 'CO', 'C_graphite', 'H2', 'O2', 'OH', 'NO', 'Al2O3', 'Al', 'NH3', 'CH4')
ELEMENT_LIST = ('C', 'H', 'N', 'O', 'Al')

def run_verification():
    print(f"{'Explosive':<15} | {'Converged':<5} | {'Major Products (mol)'}")
    print("-" * 100)
    
    # 1. Prepare Coeffs
    try:
        products_db = load_products()
        coeffs_all = []
        for s in SPECIES_LIST:
            if s in products_db:
                c = products_db[s].coeffs_high
                if c.shape[0] > 7: c = c[:7]
                coeffs_all.append(c)
            else:
                coeffs_all.append(jnp.zeros(7))
        coeffs_all = jnp.stack(coeffs_all)
    except:
        print("DB Load Failed, using zeros")
        coeffs_all = jnp.zeros((len(SPECIES_LIST), 7))
        
    # 2. Build A
    A = build_stoichiometry_matrix(SPECIES_LIST, ELEMENT_LIST)
    
    # 3. Loop
    for name, data in EXPLOSIVES.items():
        atoms_dict = data['atoms']
        atom_vec = jnp.array([atoms_dict.get(e, 0.0) for e in ELEMENT_LIST])
        
        # V input: Convert cc/g to cc/total_mass?
        # NO. The solver takes V (cm3) and Atom Vector (mol).
        # We need consistent units. P = nRT/V.
        # If V is "Specific Volume in cc/g", then atoms must be "Moles in 1g".
        # Let's calculate total mass of the atom vector.
        mw_elems = {'C': 12.01, 'H': 1.008, 'N': 14.007, 'O': 15.999, 'Al': 26.98}
        total_mass_g = sum([atoms_dict.get(e,0)*mw_elems.get(e,0) for e in ELEMENT_LIST])
        
        # If data['V_est'] is cc/g (specific volume ~ 1/rho * 0.75)
        # Total V = V_est * total_mass_g
        V_total = data['V_est'] * total_mass_g
        T = data['T_est']
        
        try:
            n_eq = solve_equilibrium(atom_vec, V_total, T, A, coeffs_all)
            
            # Format output
            # Filter small amounts
            products_str = []
            for i, s in enumerate(SPECIES_LIST):
                if n_eq[i] > 1e-3:
                    products_str.append(f"{s}:{n_eq[i]:.2f}")
            p_str = ", ".join(products_str)
            
            print(f"{name:<15} | {'✅' if not jnp.isnan(n_eq[0]) else '❌'}     | {p_str}")
            
        except Exception as e:
            print(f"{name:<15} | ❌Err  | {str(e)}")

if __name__ == "__main__":
    run_verification()
