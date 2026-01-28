import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from pdu.components import get_component
from pdu.physics.thermo import compute_total_enthalpy
from pdu.calibration.differentiable_cj_enhanced import predict_cj_with_isentrope
import numpy as np

# HMX test
comp = get_component("HMX")
density = 1.891

# Setup
from pdu.physics.database import ELEMENT_LIST, ELEMENT_ATOMIC_MASS
atomic_masses = [ELEMENT_ATOMIC_MASS[e] for e in ELEMENT_LIST]

# Formula
equiv_formula = {e: 0.0 for e in ELEMENT_LIST}
equiv_formula['C'] = comp.C * 1.0
equiv_formula['H'] = comp.H * 1.0
equiv_formula['N'] = comp.N * 1.0
equiv_formula['O'] = comp.O * 1.0
total_moles = sum(equiv_formula.values()) / 4.0  # Assume ~4 atoms per mol

atom_vec = jnp.array([equiv_formula[e] / total_moles for e in ELEMENT_LIST])
final_hof = comp.heat_of_formation / total_moles  # J/mol
final_mw = sum([atom_vec[i] * atomic_masses[i] for i in range(len(ELEMENT_LIST))])

print(f"HMX 归一化参数:")
print(f"  final_hof = {final_hof:.2e} J/mol")
print(f"  final_mw = {final_mw:.2f} g/mol")
print(f"  atom_vec (C,H,N,O,...) = {atom_vec[:6]}")

# Expected Q
Q_exp = 6.19  # MJ/kg
print(f"\n实验爆热 Q_exp = {Q_exp} MJ/kg")
print(f"预期 H_products = {final_hof - Q_exp * final_mw:.2e} J/mol")
