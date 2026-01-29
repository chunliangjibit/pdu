
import jax
import jax.numpy as jnp
import numpy as np
from pdu.physics.eos import compute_mixed_matrices_dynamic, compute_pressure_jcz3, JCZ3EOS
from pdu.api import detonation_forward

def test_ree_mixing_logic():
    print("\n=== Test 1: Dynamic Mixing Logic ===")
    T = 3000.0
    eps_vec = jnp.array([100.0, 300.0])
    r_vec = jnp.array([3.5, 3.2])
    alpha_vec = jnp.array([13.0, 13.0])
    lambda_vec = jnp.array([0.0, 1000.0]) # Species 1 has correction
    
    # Expected: 
    # eps[0] = 100
    # eps[1] = 300 * (1 + 1000/3000) = 300 * 1.333 = 400
    
    eps_mat, r_mat, alpha_mat = compute_mixed_matrices_dynamic(T, eps_vec, r_vec, alpha_vec, lambda_vec)
    
    print(f"Eps Vec Input: {eps_vec}")
    print(f"Lambda Vec: {lambda_vec}")
    print(f"T: {T}")
    print(f"Result Eps Matrix:\n{eps_mat}")
    
    # Check diagonal
    assert abs(eps_mat[0,0] - 100.0) < 1e-4
    assert abs(eps_mat[1,1] - 400.0) < 1e-4
    # Check cross
    expected_cross = np.sqrt(100.0 * 400.0) # 200
    assert abs(eps_mat[0,1] - 200.0) < 1e-4
    
    print("Logic Verification Passed!")

def test_eos_sensitivity():
    print("\n=== Test 2: EOS Pressure Sensitivity ===")
    # Create a pure H2O system simulation
    n = jnp.array([1.0])
    V = 30.0 # cc/mol ~ 0.6 g/cm3 for water
    T = 4000.0
    
    coeffs_all = jnp.zeros((1, 9)) # Dummy
    
    eps = jnp.array([333.0])
    r = jnp.array([3.27])
    alpha = jnp.array([13.0])
    
    # Case A: No Correction
    lam_A = jnp.array([0.0])
    
    # Case B: Strong Correction (Ree 1982 style)
    lam_B = jnp.array([2000.0])
    
    n = jnp.array([0.5, 0.5])
    V = 40.0
    T = 3000.0
    coeffs = jnp.zeros((2, 9))
    
    P_A = compute_pressure_jcz3(n, V, T, coeffs, eps, r, alpha, lam_A)
    print(f"P (lambda=0):    {P_A:.4f} Pa")
    
    P_B = compute_pressure_jcz3(n, V, T, coeffs, eps, r, alpha, lam_B)
    print(f"P (lambda=2000): {P_B:.4f} Pa")
    print(f"P (lambda=2000): {P_B:.4f} Pa")
    
    # Lambda increases eps -> Deeper potential -> More attraction?
    # At high T/high density, repulsion dominates.
    # Higher eps usually means "larger" effective size scaling? No.
    # Eps is energy scale. 
    # If eps increases, T*=kT/eps decreases.
    # In Exp-6, lower T* usually means stronger normalized repulsion? 
    # Let's check rep term: exp(alpha(1-r/rm)).
    # Actually, changing eps changes the attractive well depth.
    # Deeper well (higher eps) -> More attraction -> Lower Pressure (usually).
    
    diff = P_B - P_A
    print(f"Difference: {diff:.4f} Pa")
    
    if abs(diff) > 1.0:
        print("Sensitivity Confirmed. Ree Correction is Active.")
    else:
        print("WARNING: No pressure change detected!")

if __name__ == "__main__":
    test_ree_mixing_logic()
    test_eos_sensitivity()
