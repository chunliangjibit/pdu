"""
PDU Core 模块：化学平衡求解

Schur-RAND 方法 + KKT 精确梯度
"""

from pdu.core.equilibrium import solve_equilibrium, build_stoichiometry_matrix

__all__ = ['solve_equilibrium', 'build_stoichiometry_matrix']
