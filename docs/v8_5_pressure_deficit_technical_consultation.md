# PyDetonation-Ultra V8.5 技术咨询报告：爆压系统性低估与物理一致性权衡

**版本**: V8.5
**日期**: 2026-01-29
**作者**: PDU Core Team

## 1. 项目概述

### 1.1 核心目标
PyDetonation-Ultra (PDU) 旨在构建一个基于 **Jax 可微分编程** 的高保真炸药爆轰物理引擎。核心目标是实现从化学配方到爆轰性能（$D, P, T, Q$）及状态方程参数（JWL EOS）的端到端高精度预测，并支持逆向配方设计。

### 1.2 技术路线与现状
当前版本 **V8.5 (Thermodynamic Upgrade)** 已完成以下核心物理内核的升级：
- **混合精度牛顿求解器**: 实现了鲁棒的化学平衡计算。
- **JCZS3 状态方程**: 采用 Exp-6 分子势能函数描述产物相互作用。
- **热力学修正 (Phase 25)**: 实施了 $\Delta n_{gas}RT$ 修正，彻底解决了爆热 ($Q$) 计算问题。
- **物理一致性**: 将 N2/CO2/H2O 的 Exp-6 参数统一更新为基于液态 Hugoniot 数据的专家推荐值 ($\alpha=13.0, r^*$ from liquid data)。

## 2. 当前成就 (V8.5)

在实施专家反馈（V8.4 Consultation）后，我们取得了以下里程碑式的进展：

| 性能指标 | HMX (Err) | TNT (Err) | PETN (Err) | 评价 |
| :--- | :---: | :---: | :---: | :--- |
| **爆热 $Q$** | -7.9% | **+7.2%** | **+6.7%** | **(Fixed)** 之前 TNT/PETN 误差 >50%，现已完全修复。 |
| **爆速 $D$** | -10.3% | -7.6% | -9.6% | 偏差稳定，约为实验值的 90%。 |
| **爆压 $P_{CJ}$** | **-21.6%** | **-23.0%** | **-18.8%** | **(Critical)** 系统性低估，主要技术瓶颈。 |

**主要突破**: 热力学及原子索引修复 (Thermodynamic Fix) 极其成功，爆热精度已达标。

## 3. 技术瓶颈与咨询需求 (Core Bottlenecks)

### 3.1 问题描述：爆压系统性低估 (The Pressure Deficit)
在将 $N_2, CO_2, H_2O$ 的排斥参数 $\alpha$ 从之前的人工硬化值 (14.5~15.5) 回归到物理一致的专家推荐值 ($\alpha=13.0$) 后，所有炸药的 **爆压 $P_{CJ}$ 出现了约 20% 的系统性低估**。

*   **HMX 实测**: 42.0 GPa $\rightarrow$ **V8.5 预测**: 32.9 GPa
*   **RDX 实测**: 34.7 GPa $\rightarrow$ **V8.5 预测**: 29.8 GPa

### 3.2 根本原因分析 (Root Cause)
1.  **势能函数过软**: $\alpha=13.0$ 对应的排斥力较弱。虽然这符合纯液态组分的 Hugoniot 数据（中低压段），但在爆轰波阵面的极高压环境 (>30 GPa) 下，分子间排斥力可能需要更强的硬化项。
2.  **混合规则局限**: 当前使用理想的单流体混合规则 (One-Fluid Conformal Solution)。在极高压下，不同尺寸分子 ($N_2$ vs $H_2O$) 的非理想混合效应可能显著增加了体系的压力（体积排斥效应），而当前模型未捕捉到这一点。
3.  **固相状态方程**: 碳和氧化铝采用 Murnaghan EOS，其参数是否需要在纳秒级爆轰过程中进行动态调整（如考虑相变滞后或纳米尺寸效应）尚不确定。

### 3.3 咨询问题 (Consultation Questions)

> [!CAUTION]
> **核心矛盾**: 我们不希望通过简单的“凑参数”（如人为调高 $\alpha$）来拟合数据，因为这会破坏物理一致性。

**Q1. 高压排斥项修正策略 (P0)**:
在保持 $\alpha=13.0$ (符合 Liquid Hugoniot) 的物理基准下，是否有标准的物理修正项（如由许多文献提到的 $\Delta \phi_{excess}$ 或交叉势能修正）可以用来补偿高压下的压力缺失？

**Q2. 混合规则优化 (P1)**:
当前的 JCZ3 单流体近似是否是低压的主要原因？是否建议引入改进的混合规则（如 Ree Van Thiel 混合律或其他修正因子 $\lambda$）？

**Q3. JWL 参数映射 (P2)**:
我们观察到 V8.5 导出的 JWL $A$ 参数普遍偏低（与 $P_{CJ}$ 一致）。如果 $P_{CJ}$ 无法物理地提高，我们是否应该在 JWL 拟合阶段引入 "Pressure Bias" 来强行对齐工程应用需求？

## 4. 代码库状态

*   **仓库地址**: `chunliangjibit/pdu.git`
*   **当前分支**: `main`
*   **最新提交**: `feat: V8.5 Thermodynamic Upgrade - Fixed Q calc and updated JCZS3 EOS`
*   **关键文件**:
    *   `pdu/data/jcz3_params.json`: 包含当前的 $\alpha=13.0$ 参数集。
    *   `docs/v8_5_performance_report.md`: 完整的 5 参数验证数据。

## 5. 优先级排序表

| ID | 任务名称 | 优先级 | 描述 |
| :--- | :--- | :---: | :--- |
| **Task-01** | **Pressure Recovery Theory** | **P0** | 寻找物理上合理的压力补偿机制 (非单纯调参)。 |
| **Task-02** | **Mixing Rule Check** | P1 | 评估混合规则对 $P_{CJ}$ 的贡献。 |
| **Task-03** | **Solid EOS Calibration** | P2 | 检查碳/氧化铝 EOS 参数对混合炸药压力的影响。 |

## 6. 附录

*   [V8.5 性能对标报告](./v8_5_performance_report.md)
*   [JCZS3 参数文件](../pdu/data/jcz3_params.json)
