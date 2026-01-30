import jax
import jax.numpy as jnp
import time
from pdu.core.equilibrium import solve_equilibrium, build_stoichiometry_matrix
from pdu.utils.precision import R_GAS

# Important: Enable FP64 for verification
jax.config.update("jax_enable_x64", True)

def test_epm_solver():
    print("Testing EPM Solver (Phase 2)...")
    
    # 1. Setup a standard CHNO system
    species = ['N2', 'CO2', 'H2O', 'CO', 'H2', 'O2']
    elements = ['C', 'H', 'N', 'O']
    A = build_stoichiometry_matrix(species, elements)
    # Add a dummy 5th element to match 5x5 (Schur-KKT constraint)
    A = jnp.vstack([A, jnp.zeros((1, A.shape[1]))])
    elements.append('Dummy')
    
    print(f"Elements: {elements}")
    
    # Realistic dummy coeffs: H ~ -300kJ/mol, S ~ 200J/molK
    # G/RT ~ (-300e3 - 3000*200) / (8.314 * 3000) ~ -36
    h_dummy = -300000.0 * jnp.ones((len(species),))
    s_dummy = 200.0 * jnp.ones((len(species),))
    
    # NASA 9-coeffs has a specific structure. pdu/physics/thermo.py handles it.
    # coeffs[2] is the constant cp term, coeffs[7] is H/R, coeffs[8] is S/R
    coeffs = jnp.zeros((len(species), 9))
    coeffs = coeffs.at[:, 2].set(4.0) # Cp/R ~ 4
    coeffs = coeffs.at[:, 7].set(-30000.0) # H/R
    coeffs = coeffs.at[:, 8].set(20.0) # S/R
    
    atom_vec = jnp.array([1.0, 4.0, 2.0, 6.0, 0.0]) # C1 H4 N2 O6
    V = 50.0 # cm3
    T = 3000.0 # K
    eos_params = (jnp.zeros(6), jnp.zeros(6), jnp.zeros(6), jnp.zeros(6), None, None) 
    
    # 2. Test Forward
    print("Running Forward Solve...")
    start = time.time()
    n_star = solve_equilibrium(atom_vec, V, T, A, coeffs, eos_params)
    jax.block_until_ready(n_star)
    print(f"  Forward Success! Time: {time.time()-start:.4f}s")
    print(f"  n_star: {n_star}")
    print(f"  Mass Balance Check (res): {jnp.linalg.norm(A @ n_star - atom_vec)}")
    
    # 3. Test Backward (Gradient)
    print("Running Backward (Grad) Solve...")
    def loss_fn(atom):
        n = solve_equilibrium(atom, V, T, A, coeffs, eos_params)
        return jnp.sum(n**2)
        
    grad_fn = jax.grad(loss_fn)
    start = time.time()
    g_atom = grad_fn(atom_vec)
    jax.block_until_ready(g_atom)
    print(f"  Backward Success! Time: {time.time()-start:.4f}s")
    print(f"  Grad w.r.t atom_vec: {g_atom}")

if __name__ == "__main__":
    test_epm_solver()


if __name__ == "__main__":
    test_epm_solver()
