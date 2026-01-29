# PyDetonation-Ultra V8.6 技术咨询报告：JWL 拟合策略与含铝炸药非理想性

**版本**: V8.6 (Stiff-Mix Protocol)
**日期**: 2026-01-29
**密级**: 内部技术交流

## 1. 项目全景与当前物理态势 (Project Overview)

### 1.1 系统架构
PyDetonation-Ultra (PDU) 是基于 **Jax 框架** 构建的下一代含能材料物理引擎。其核心特征包括：
- **完全可微分 (Fully Differentiable)**: 支持端到端的梯度反向传播，专为逆向配方设计（Inverse Design）打造。
- **混合精度求解器**: 结合 Float32 的计算速度与 Float64 的热力学精度，实现了鲁棒的 KKT 化学平衡求解。
- **物理内核 V8.6**: 采用 "Stiff-Mix" 协议，包括：
    - **气相**: JCZS3 (Exp-6) 状态方程，参数基于液态 Hugoniot 数据 ($\alpha=13.0$) 并附加高压斥力修正 ($\Delta F_{rep}$)。
    - **固相**: 碳采用 Fried-Howard 液态碳模型，氧化铝采用 Murnaghan 方程。
    - **混合规则**: 目前为 vdW1f 单流体近似，但在自由能计算中实施了严格的气固体积扣除。

### 1.2 物理一致性与工程精度的统一
在 V8.6 中，我们成功解决了“爆压系统性低估”问题。通过引入 **液态碳大体积效应** 和 **高密度硬化项**，核心单体炸药（HMX, RDX, TNT, PETN）的爆压 ($P_{CJ}$) 预测误差已收敛至 **5% 以内**，同时保持了爆热 ($Q$ < 10%)。这证明了我们的热力学修正 ($\Delta n_{gas}RT$) 对于 CHNO 理想炸药是完全正确的。

## 2. JWL 状态方程拟合策略 (Current Strategy)

PDU 的核心输出并非仅是 CJ 点参数，而是完整的 **JWL 状态方程系数** ($A, B, R_1, R_2, \omega$)，以供 LS-DYNA 等工程软件使用。

### 2.1 拟合流程
1.  **生成等熵线 (Isentrope)**: 从 CJ 点开始，沿等熵路径 ($S=const$) 膨胀至 $V/V_0 = 10.0$。在每一点求解化学平衡，获得 $(V, P, E)$ 数据对。
2.  **工程偏置拟合 (Engineering Bias)**: 为了保证工程实用性，我们在拟合损失函数中引入了强约束（Lagrange Multiplier）：
    - 强制 JWL 曲线通过实验 CJ 点 $(V_{CJ}^{exp}, P_{exp})$。
    - 这确保了即便物理计算的 $P_{CJ}$ 略有偏差，输出的 JWL 参数也能正确反映炸药的各项同性冲击阻抗。

### 2.2 CHNO 炸药的表现
对于理想炸药，该策略表现优异。以 **HMX** 为例，V8.6 输出参数与 LLNL 标准参数及实验值高度吻合，相对误差 < 5%。

## 3. 核心瓶颈：含铝炸药的“非理想”异常 (The Aluminized Anomaly)

然而，在处理含铝/含粘结剂的混合炸药（特别是 **PBXN-109**: RDX/Al/Binder）时，我们遇到了巨大的误差，这表明当前的“理想平衡”假设可能不再适用。

### 3.1 误差数据 (PBXN-109)
| 参数 | PDU V8.6 预测值 | 实验/标准值 | 相对误差 |
| :--- | :---: | :---: | :---: |
| 爆速 $D$ | 7963 m/s | 7600 m/s | +4.8% (Acceptable) |
| 爆压 $P_{CJ}$ | 21.9 GPa | 22.0 GPa | -0.6% (Excellent) |
| **爆热 $Q$** | **11.78 MJ/kg** | **7.20 MJ/kg** | **+63.7% (Critical!)** |
| JWL $A$ | 1224 GPa | 758 GPa | **+61.4%** |
| JWL $B$ | 15.6 GPa | 8.7 GPa | **+80.0%** |

### 3.2 现象深度剖析：能量过剩危机 (The Energy Crisis)
PBXN-109 的爆热误差 (+63.7%) 并非简单的数值偏差，而是**热力学假设崩溃**的体现：

1.  **理想氧化假设 vs 动力学现实**: 
    PDU 求解器基于 $\Delta G_{min}$ 原理，强制所有铝粉在 CJ 面瞬间与产物 ($H_2O, CO_2$) 发生置换反应生成 $Al_2O_3$。
    $$2Al + 3H_2O \rightarrow Al_2O_3 + 3H_2 + \Delta H_{huge}$$
    而在真实爆轰波阵面 (Reaction Zone ~mm级)，铝粉由于氧化层阻滞，反应度往往不足 50%。大量的铝是在 CJ 面后的膨胀区才开始缓慢燃烧 (After-burning)。

2.  **状态方程的双重打击**:
    - **V8.6 修正的副作用**: 为了修复理想炸药的爆压，我们引入了 "$C=15.0$" 的高压斥力修正。
    - 在 PBXN-109 中，产生的固体 $Al_2O_3$ 被强制扣除体积，进一步压缩了气相空间，导致气相自由能急剧升高。
    - **高能 + 高压**: 求解器为了把这部分巨大的化学能“排泄”掉，被迫在膨胀等熵线上通过极高的 $P(V)$ 来做功，直接导致 JWL 的 $A/B$ 参数暴涨至物理上荒谬的数值 (A=1224 GPa)。

## 4. 咨询需求与技术探讨 (Consultation Topics)

我们需要针对含铝炸药的 JWL 建模策略寻求专家指导：

### Q1: 铝反应的非理想性截断 (Non-Ideal Cutoff)
实验观测表明，铝粉通常在 CJ 面后发生显著的二次反应（After-burning）。PDU 作为平衡代码，是否应该在 CJ 计算中人为限制铝的反应度？
- **策略 A**: 假设铝在 CJ 点仅作惰性热汇，不参与反应？
- **策略 B**: 引入经验性的 "Reacted Fraction" $\lambda$ (e.g. $\lambda_{Al}=0.3$)？

### Q2: 针对“圆筒实验”的拟合目标
实验测得的 $Q$ (7.2 MJ/kg) 往往是特定体积膨胀比下的做功量。
- 我们是否应该放弃拟合“理论全平衡等熵线”，转而让 JWL 去拟合一个能量被截断的“有效等熵线”？
- 如果是，如何定义这个截断能？

### Q3: 推荐的工程化策略
如果无法在物理层面模拟复杂的非理想动力学，是否有数学上的 Trick 能生成一套“爆压正确、能量也正确”的 JWL 参数？
- 例如：强行修改配方输入，减少铝含量？
- 或者在 JWL 拟合时，不再锚定理论等熵线，而是锚定标准圆筒实验的数据点？

## 6. 附录：核心代码与关键参数 (Code & Configuration)

### 6.1 爆热与内能计算逻辑 (pdu/api.py)
V8.6 严格遵循内能守恒，使用了 `final_internal_energy` 进行 Q 计算。

```python
# [pdu/api.py] Heat of Detonation Calculation
# 1. Reactant Internal Energy (Corrected for Delta n_gas RT)
E0 = final_internal_energy 

# 2. Product Internal Energy (Standard State U = H - PV)
Uf_CO2 = Hf_CO2 - 1.0 * R * T  # Gas correction
Uf_H2O = Hf_H2O - 1.0 * R * T  # Gas correction
Uf_Al2O3 = Hf_Al2O3            # Solid (No PV correction)

# 3. Q Calculation
Q = (final_internal_energy - U_prod) / Mass
```

### 6.2 状态方程修正协议 (pdu/physics/eos.py)
V8.6 的 "Stiff-Mix" 协议包含两个关键修正：

```python
# [pdu/physics/eos.py] V8.6 Stiff-Mix Upgrade

# A. High-Pressure Repulsive Correction (Stiffening)
# C_stiff = 15.0 calibrated to HMX P_CJ = 41.4 GPa
A_rep = n_gas_total * R * T * 15.0 * (eta**4)
A_gas_total = A_gas_ideal + A_excess_hs + U_attr + A_rep

# B. Fried-Howard Liquid Carbon EOS
# Provides larger solid volume -> Increased gas compression
c_compress = jnp.power(1.0 + 6.0 * P_est / 60e9, -1.0/6.0)
vol_c = 4.44 * c_compress # V0 = 4.44 cc/mol (Liquid)
```

### 6.3 物理一致性参数集 (jcz3_params.json)
即使为了追求爆压，我们依然坚持了 $\alpha=13.0$ 的液态 Hugoniot 基准，没有进行人为参数回调。

```json
{
  "N2": {
    "note": "JCZS3 Expert Calibration (Liquid N2 Hugoniot)",
    "epsilon_over_k": 103.0,
    "r_star": 4.11,
    "alpha": 13.0
  },
  "CO2": {
    "note": "JCZS3 Expert Calibration (Liquid CO2 Hugoniot)",
    "epsilon_over_k": 240.0,
    "r_star": 4.30,
    "alpha": 13.0
  }
}
```

## 5. 结论
PDU V8.6 在理想炸药领域已臻化境，但在非理想（含铝）领域仍面临“过于理想化”带来的模型失真。我们急需一套针对含铝体系的 **JWL 修正策略**，以平抑过剩的计算能量，输出具备工程实用价值的参数。
