# HMX V11 标定稳定性与“冷发散”问题技术咨询需求 (V1.2)

> **参考技能 (Skills Used)**: 
> - [写作技能 (Writing Skills)](file:///home/jcl/HDY/PyDetonation-Ultra/.agent/skills/antigravity-awesome-skills/skills/writing-skills/SKILL.md) / [Anthropic Best Practices](file:///home/jcl/HDY/PyDetonation-Ultra/.agent/skills/antigravity-awesome-skills/skills/writing-skills/anthropic-best-practices.md) (核心原则：简洁、结构化方案)
> - [计划写作 (Plan Writing)](file:///home/jcl/HDY/PyDetonation-Ultra/.agent/skills/antigravity-awesome-skills/skills/plan-writing/SKILL.md) (任务分解与目标对齐)

## 1. 背景与进展综述

针对此前专家咨询意见（[回复.md](file:///home/jcl/HDY/PyDetonation-Ultra/documents/回复.md)）中提到的 Patch A-D 方案，项目组已完成初步实施。HMX V11 标定任务目前状态如下：

- **[已修复] 反应停滞 (Stagnation)**：通过统一 `implicit_eos.py` 中的 SI/CGS 单位链，目前 ZND 能够成功点火，初始压力 $P_0$ 恢复正常。
- **[已修复] 压力异常 (992 GPa Anomaly)**：确认为产物组分默认 $r^*$ (3.5 Å) 过大导致的超高压过饱和。强制调整至 **3.1 Å** 后，VN Spike 压力降至 **57.17 GPa**（物理合理区间）。
- **[已实施] 鲁棒积分器 (Reject & Shrink)**：`znd.py` 已具备坏点检测与步长回退能力，解决了 NaN 崩溃与日志洪泛问题。

---

## 2. 当前核心瓶颈：冷发散 (Cold Divergence)

在 HMX ZND 剖面计算中，出现了严重的“膨胀制冷”主导现象。

### 现象描述：
1. **温度骤降**：在 VN 点（~4500 K）过后，由于强烈的机械膨胀做功，气体温度 $T$ 迅速下跌，跌破 200 K 甚至到达 140 K 左右。
2. **物理反馈失效**：由于温度极低，JCZ3 模型中的有效直径比率 $d/r^*$ 剧增（触发了 $d/r^*$ 的光滑地板），导致即使在较低密度下，填充率 $\eta$ 也会重新飙升。
3. **输出发散**：反应末端（$\lambda \rightarrow 1$）压力出现 `inf`（无穷大），或者步长被无限压缩至失效。

### 已尝试的对冲措施：
- **大幅提升反应速率**：将动力学参数 $k_p$ 从 10.0 提升至 1000.0，试图通过强化化学放热功率来对抗膨胀制冷。
- **观测结果**：依然无法稳定维持 $T > 1000$ K，反应区末尾的压力不一致性依然存在。

---

## 3. 需咨询的技术问题

针对上述“冷发散”导致的数值/物理不稳定，需请专家协助明确以下方向：

### Q1: 动力学强度与物理真实性的平衡
$k_p = 1000.0$ 是否已经大幅偏离了 HMX 的物理尺度？在可微 ZND 框架下，是否存在某种物理机制（如：特定的热沉模型或反应波结构调节）能更自然地抑制这种超低温发散？

### Q2: JCZ3 模型在极低温下的适用性
JCZ3 模型在 $T < 500$ K 的超高压环境下，其有效直径表达式是否存在失效风险？是否需要引入温度下界保护或针对膨胀区的势能修补？

### Q3: 能量守恒项的数值刚性
当前 `dT/dxi = (放热 - 膨胀功) / Cv`。在膨胀极快时，这一项极其敏感。是否建议对积分路径进行准静态化处理，或者在 ZND 场方程中引入额外的正则化项以平滑温度演化？

### Q4: 产物 $r^*$ 与 $P_{CJ}$ 的耦合调节
目前为了解决 992 GPa 异常，将 $r^*$ 硬调为 3.1 Å。这是否会由于排斥势变弱，导致后续 CJ 状态无法对齐实验数据？在 V11 多相流过渡期，是否有更系统的参数对齐建议？

---

## 4. 交付要求

- 请针对上述问题提供理论解释或 Patch E 方案建议。
- 若涉及代码逻辑修改，请提供类似“回复.md”中的 pseudocode 片段。

**文档保存路径**: `/home/jcl/HDY/PyDetonation-Ultra/documents/HMX_V11_Consultation_Request.md`
