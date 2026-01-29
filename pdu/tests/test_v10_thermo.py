
import jax
import jax.numpy as jnp
from pdu.physics.thermo import compute_cp, compute_enthalpy, compute_entropy
from pdu.data.products import get_product_thermo

def verify_nasa9_dispatch():
    print("\n=== Test: NASA-7 vs NASA-9 Dispatch ===")
    
    # Reload H2O from the database (now should have NASA-9)
    h2o = get_product_thermo("H2O")
    
    T = 3000.0
    
    # 1. Manual extraction of coeffs for testing
    c7_high = h2o.coeffs_high
    c9_high = h2o.coeffs_high_9
    
    print(f"Species: {h2o.name} at T={T}K")
    print(f"C7 Coeffs Len: {len(c7_high)}")
    print(f"C9 Coeffs Len: {len(c9_high)}")
    
    # 2. Compute properties using automated dispatch
    # Since h2o.get_coeffs(T) will now return C9 if available
    cp_auto = compute_cp(h2o.get_coeffs(T), T)
    h_auto = compute_enthalpy(h2o.get_coeffs(T), T)
    s_auto = compute_entropy(h2o.get_coeffs(T), T)
    
    # 3. Compute properties using manual C7 (to see difference)
    cp_c7 = compute_cp(c7_high, T)
    h_c7 = compute_enthalpy(c7_high, T)
    s_c7 = compute_entropy(c7_high, T)
    
    print(f"\nResults Comparison (C7 vs C9_Auto):")
    print(f"Cp (J/mol-K): C7={cp_c7:.4f}, C9={cp_auto:.4f}, Diff={abs(cp_auto-cp_c7):.4f}")
    print(f"H  (J/mol):   C7={h_c7:.4f}, C9={h_auto:.4f}, Diff={abs(h_auto-h_c7):.4f}")
    print(f"S  (J/mol-K): C7={s_c7:.4f}, C9={s_auto:.4f}, Diff={abs(s_auto-s_c7):.4f}")
    
    # 4. Success criteria: C9 should be slightly different but in same ballpark
    assert abs(cp_auto - cp_c7) / cp_c7 < 0.1 # < 10% diff
    print("\nNASA-9 Dispatch and Calculation Successful!")

if __name__ == "__main__":
    verify_nasa9_dispatch()
