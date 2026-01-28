"""
PyDetonation-Ultra (PDU)
========================

基于 JAX 的可微分爆轰物理计算框架。

主要功能：
- 正向计算：配方 → 爆轰性能 (D, P, T, JWL)
- 逆向设计：目标性能 → 最优配方
"""

__version__ = "0.1.0"
__author__ = "PDU Development Team"

# 环境配置
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".85")

# 延迟导入，避免循环依赖
def __getattr__(name):
    if name == "detonation_forward":
        from pdu.api import detonation_forward
        return detonation_forward
    elif name == "inverse_design":
        from pdu.api import inverse_design
        return inverse_design
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "detonation_forward",
    "inverse_design",
    "__version__",
]
