# pdu/tests/grid_search_hmx.py
import jax
import jax.numpy as jnp
import os
import sys

# Ensure PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pdu.tests.sweep_hmx_params import run_hmx_point

def main():
    print(f"{'r_sc':<6} | {'q_fac':<6} | {'P_CJ':<6} | {'T_CJ':<6}")
    print("-" * 35)
    for r_sc in [0.80, 0.82, 0.84, 0.86]:
        for q_fac in [0.55, 0.65, 0.75]:
            q_val = 5.8e6 * q_fac
            p, t = run_hmx_point(r_sc, 13.5, q_val)
            if p:
                print(f"{r_sc:<6.2f} | {q_fac:<6.2f} | {p:<6.2f} | {t:<6.0f}")
            else:
                print(f"{r_sc:<6.2f} | {q_fac:<6.2f} |  FAIL  |  FAIL")

if __name__ == "__main__":
    main()
