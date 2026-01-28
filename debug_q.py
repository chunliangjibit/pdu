import jax
jax.config.update("jax_enable_x64", True)
from pdu.api import detonation_forward

# Quick debug test for HMX
res = detonation_forward(["HMX"], [1.0], 1.891, verbose=False)
print(f"HMX Q = {res.Q:.2f} MJ/kg (expected: 6.19 MJ/kg)")
print(f"Error: {(res.Q - 6.19)/6.19*100:.1f}%")
