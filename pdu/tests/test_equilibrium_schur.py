
import jax
import jax.numpy as jnp
import numpy as np
from pdu.core.equilibrium import solve_equilibrium, build_stoichiometry_matrix
from pdu.data.products import load_products
from pdu.utils.precision import R_GAS

def test_equilibrium_hmx():
    print("Testing Schur-RAND Equilibrium Solver...")
    
    # 1. Setup Mock Data
    atom_vec = jnp.array([4.0, 8.0, 8.0, 8.0]) # C, H, N, O
    
    # Species list (Simplified subset for testing)
    species_list = ('CO2', 'H2O', 'N2', 'CO', 'H2', 'O2', 'NO', 'OH')
    # Elements order must match atom_vec
    element_list = ('C', 'H', 'N', 'O')
    
    # Build A matrix explicitly
    A = build_stoichiometry_matrix(species_list, element_list)
    
    # Load coeffs (Mock or Real)
    # Try to load real products if possible
    try:
        products_db = load_products()
        coeffs_all = []
        for s in species_list:
            if s in products_db:
                # Assuming products_db[s].coeffs_high matches expected shape (7)
                # If DB has 9 (including integration constants), we might need to slice
                # But let's assume valid data for now, or just use zeros for test
                c = products_db[s].coeffs_high
                if c.shape[0] > 7:
                    c = c[:7] # Take first 7 if it happens to be 9
                coeffs_all.append(c) 
            else:
                coeffs_all.append(jnp.zeros(7))
        coeffs_all = jnp.stack(coeffs_all)
    except Exception as e:
        print(f"Failed to load real DB: {e}")
        coeffs_all = jnp.zeros((len(species_list), 7)) # Dummy
    
    V_input = 120.0 # cm3
    T_input = 4000.0 # K
    
    print(f"Input: Atoms={atom_vec}, V={V_input}, T={T_input}")
    
    # 2. Test Forward
    try:
        n_eq = solve_equilibrium(
            atom_vec, V_input, T_input, 
            A, coeffs_all
        )
        print("Forward Result (Moles):")
        for s, n in zip(species_list, n_eq):
            print(f"  {s}: {n:.4f}")
            
        # Check conservation
        b_calc = A @ n_eq
        err = jnp.linalg.norm(b_calc - atom_vec)
        print(f"Conservation Error: {err:.2e}")
        
    except Exception as e:
        print(f"❌ Forward Execution Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Test Backward (Gradients)
    print("\nTesting Gradients (dL/dAtoms)...")
    
    def loss_fn(atoms):
        n = solve_equilibrium(
            atoms, V_input, T_input,
            A, coeffs_all
        )
        # Dummy loss: Maximize N2 (index 2)
        idx_n2 = species_list.index('N2')
        return -n[idx_n2]
    
    try:
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(atom_vec)
        print("Gradient vector:", grads)
        
        if jnp.all(jnp.isfinite(grads)):
            print("✅ Gradient Computation Passed")
        else:
            print("❌ Gradient Computation Failed (NaN/Inf)")
            
    except Exception as e:
        print(f"❌ Gradient Execution Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_equilibrium_hmx()
