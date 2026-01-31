# PyDetonation-Ultra (PDU) 开发规则

## **CRITICAL PROTOCOL: SKILL-FIRST WORKFLOW (MANDATORY)**
**RULE #1:** Whenever you encounter a problem, need to implement a feature, or plan a task, you **MUST FIRST** search the `/home/jcl/HDY/PyDetonation-Ultra/.agent/skills` directory.
- **Action**: Use `find_by_name` or `grep_search` to find relevant skills.
- **Integration**: Explicitly cite the skill you are using in your plan and follow its instructions.
- **Failure to check skills first is a violation of project rules.**

## 0. 项目目标与核心功能

### 0.1 正向计算：全面爆轰性能预测
**输出参数**：D, P_CJ, T_CJ, Q, JWL-5参数 (A, B, R1, R2, w).

### 0.2 [已归档] 逆向设计
(注：逆向配方设计模块已在 V10.6.1 中剔除，以聚焦于 V11 两相流动力学引擎的开发。架构上仍保持算子可微性。)

---

## 2. 代码架构原则

### 2.1 精度控制策略
- **全局启用 Float64**: `jax.config.update("jax_enable_x64", True)`。

---

## 4. 物理正确性原则

### 4.5 V10.6 物理与拟合强制规范 (V10.6 Mandates)
- **Matrix Quenching**: 对于含金属（Al, B, etc.）混合炸药，必须在 `api.py` 中应用矩阵淬灭因子（建议 0.94-0.98），模拟反应效率下降。
- **Physical Heat Sink**: 惰性组分必须被建模为热沉，在 CJ 平衡计算中扣除其升温显热。
- **JWL Physics Barriers**: 
    - **Grüneisen Range**: 强制 $\omega \in [0.25, 0.45]$。
    - **Topology Ratio**: 强制 $B/A < 0.1$。
    - **Energy Cutoff**: 含铝炸药 JWL $E_0$ 目标值需乘上有效做功系数（建议 0.72），将后燃烧能量剥离。
    - **Slope Anchor**: 拟合必须考虑 CJ 点的 Rayleigh 线斜率约束。

### 4.6 V11.0 Engineering & Diagnostic Protocols
- **Triple Output Heat Capacity**: 所有的热力学诊断输出必须同时包含：总量 ($C_v$, J/K)、质量比比热 ($c_v$, J/kg/K) 和摩尔比热 ($\bar c_v$, J/mol/K)。
- **Asymptotic Ideal Gas Limit**: 新的 EOS 修改必须通过低密度压力自检（建议 $V=1000\text{ cm}^3\text{/mol}$ 时误差 $< 3\%$）。
- **Reference State Anchoring**: 所有自由能修正项（如吸热修正）必须确保在 $T_{ref}=298.15\text{ K}$ 处对原始自由能零点无干扰 ($A_{add}=0$)。
- **Domain Protection**: 对于存在热力学不稳定区（$C_v < 0$）的模型，必须在代码中添加显式的 Domain Stability Notice。

---

## 5. 测试与验证规范

### 5.2 物理验证协议
必须运行全量对标脚本，并将结果更新至 `docs/project_whitepaper.md`。核心热力学一致性验证需通过 `pdu/tests/test_thermo_consistency.py`。

### 5.4 诚实原则 (Honest Disclosure)
- 严禁隐瞒 JWL 参数的非物理行为（如 B < 0）。
- 必须如实报告含铝炸药的爆压/爆速偏差，作为模型演进的动力。

---
**版本**: V11.0 "Multi-Phase Dynamics Transition"
**更新日期**: 2026-01-31
