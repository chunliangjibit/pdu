"""PDU Physics Module - 物理模块"""

from pdu.physics.thermo import compute_gibbs, compute_enthalpy, compute_entropy
from pdu.physics.eos import compute_pressure_jcz3, JCZ3EOS, compute_chemical_potential_jcz3
from pdu.physics.potential import exp6_potential
from pdu.physics.jwl import fit_jwl_from_isentrope as fit_jwl, JWLParams
from pdu.physics.sensitivity import estimate_impact_sensitivity, compute_oxygen_balance

__all__ = [
    "compute_gibbs",
    "compute_enthalpy", 
    "compute_entropy",
    "compute_pressure",
    "JCZ3EOS",
    "exp6_potential",
    "fit_jwl",
    "JWLParams",
    "estimate_impact_sensitivity",
    "compute_oxygen_balance",
]
