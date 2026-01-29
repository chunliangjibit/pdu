import numpy as np
import jax.numpy as jnp
from pdu.api import detonation_forward

def debug_tritonal():
    print("=== Debugging Tritonal V10 ===")
    components = ['TNT', 'Al']
    fractions = [0.8, 0.2]
    rho = 1.72
    
    # V9 Partial Reaction Params for Tritonal
    # degree=1.0 (Full reaction for testing)
    res = detonation_forward(
        components, fractions, rho,
        reaction_degree={'Al': 1.0},
        verbose=True,
        fitting_method='PSO'
    )
    
    print(f"\nD_cj: {res.D}")
    print(f"P_cj: {res.P_cj}")
    print(f"T_cj: {res.T_cj}")

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    debug_tritonal()
