# PyDetonation-Ultra (PDU)

基于 JAX 的可微分爆轰物理计算框架。

## 功能

- **正向计算**：配方 → 爆速、爆压、爆温、JWL 参数等
- **逆向设计**：目标性能 → 最优配方

## 安装

```bash
mamba activate nnrf
pip install -e .
```

## 快速开始

```python
from pdu import detonation_forward, inverse_design

# 正向计算
result = detonation_forward(
    components=["RDX", "TNT"],
    fractions=[0.6, 0.4],
    density=1.75
)

print(f"爆速: {result.D:.0f} m/s")
print(f"爆压: {result.P_cj:.1f} GPa")
print(f"JWL: A={result.jwl.A:.2f}, B={result.jwl.B:.2f}")

# 逆向设计
recipe = inverse_design(
    targets={"D": 8500, "P_cj": 32},
    available_components=["RDX", "HMX", "TNT", "Al"]
)
```

## 许可证

MIT
