import jax
import jax.numpy as jnp
from pdu.api import detonation_forward

def test_nm_improvement():
    print("Testing V10.1 Physical Core for NM (Nitromethane)...")
    # Exp: P=12.6 GPa, D=6260 m/s, T=3600 K
    res = detonation_forward(['NM'], [1.0], 1.128, verbose=True)
    
    print(f"\nNM Results:")
    print(f"P_cj: {res.P_cj:.2f} GPa (Target 12.6, V10.0 was 11.2)")
    print(f"D_cj: {res.D:.0f} m/s (Target 6260)")
    print(f"T_cj: {res.T_cj:.0f} K (Target 3600, V10.0 was 3047)")

if __name__ == "__main__":
    test_nm_improvement()
