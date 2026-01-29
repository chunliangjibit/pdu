---
name: Generate PDU Consultation Whitepaper
description: Runs the full PDU benchmark suite and updates the technical consultation whitepaper with the latest results, code snippets, and error analysis.
---

# PDU Consultation Whitepaper Generation Skill

This skill automates the process of validating PyDetonation-Ultra (PDU) physics performance and generating a comprehensive technical whitepaper for expert consultation.

## Workflow

### 1. Execute Active Benchmark Suite
Identify and run the current comprehensive benchmark test script (e.g., matching `pdu/tests/test_v*_benchmark.py`) to generate fresh performance data.

**Command Reference** (Adjust filename as version evolves):
```bash
cd /home/jcl/HDY/PyDetonation-Ultra
# Find the latest benchmark file, e.g., test_v10_3_benchmark.py
PYTHONPATH=. python pdu/tests/test_v10_3_benchmark.py
```

### 2. Data & Verification Requirements (CRITICAL)
When generating the report, you MUST ensure the following data is extracted and presented with **percentage errors** for every single parameter.

**Required Explosives List (Full Spectrum):**
1.  **Pure & Liquid**: HMX, RDX, PETN, TNT, NM (Nitromethane)
2.  **Mixtures**: Comp B, Octol
3.  **Aluminized (Non-Ideal)**: Tritonal, PBXN-109

**Required Parameters (Full Calibration):**
For EACH explosive, you must report:
*   **Detonation Performance**:
    *   $D$ (Velocity)
    *   $P_{CJ}$ (Pressure)
    *   $T_{CJ}$ (Temperature)
    *   $Q$ (Heat of Detonation)
*   **JWL EOS Parameters**:
    *   $A, B$ (Pressure Terms)
    *   $R_1, R_2$ (Decay Rates)
    *   $\omega$ (Grüneisen Parameter)

**Format Requirement**:
*   Every cell must show: `Predicted / Experimental (Error%)`
*   Example: `34.0 / 34.7 (-2.2%)`

### 3. Gather Code Context
Read the following core files to extract the latest implementation details for the Whitepaper:
    *   `pdu/api.py` -> `detonation_forward`
    *   `pdu/core/equilibrium.py` -> `solve_equilibrium`
    *   `pdu/physics/eos.py` -> `compute_polar_epsilon_ree_ross`
    *   `pdu/physics/kinetics.py` -> **ALL** new kinetic functions (Miller V3, Cap, Da, Freeze, etc.)
    *   `pdu/physics/jwl.py` -> `fit_jwl_from_isentrope`

### 4. Generate Whitepaper
Update or recreate `docs/project_whitepaper.md` following the STRICT format below.

**Critical Requirements:**
*   **Full Data**: Must include ALL 9 explosives.
*   **Error Statistics**: You must manually calculate/verify which parameters have >10% error based on the new results.
    *   **Scope**: Includes Detonation Performance (D, P, T, Q) AND **JWL Parameters (A, B, R1, R2, w)**.
    *   🔴 **Red Critical**: Error > 30% (Use [!IMPORTANT] alert for major physics issues)
    *   ⚠️ **Yellow Warning**: Error 10% - 30%
*   **Consultation Questions (CRITICAL)**:
    *   You MUST consolidate **ALL** parameters with >10% error (both Detonation & JWL) into a dedicated "Consultation Questions" section.
    *   Format: "Why does [Explosive] [Parameter] deviate by [Error%]? Is the reference data constrained differently?"
    *   Do not leave any >10% error un-addressed in this question list.

## Whitepaper Template

```markdown
# PyDetonation-Ultra (PDU) V10.3 技术咨询白皮书

**版本**: V10.3 "Miller-PDU V3" (Update with current version)
**日期**: YYYY-MM-DD
**目的**: 专家咨询参考文档

---

## 一、项目概述

### 1.1 项目简介
[Insert Project Description: Based on JAX, Physics-First, targets CJ & JWL prediction, supported explosives list]

### 1.2 项目结构
[Insert Directory Tree representation]

### 1.3 核心改进
| 模块 | 改进内容 | 文件 |
|:---|:---|:---|
| [Module] | [Feature] | [File] |

---

## 二、核心代码解析

### 2.1 主入口函数: detonation_forward()
**文件**: `pdu/api.py`
```python
[Insert latest signature and docstring]
```

### 2.2 化学平衡求解器
**文件**: `pdu/core/equilibrium.py`
```python
[Insert latest signature and docstring]
```

### 2.3 [Other Key Functions...]
[Include EOS, Kinetics (Miller V3, Cap, Da, Freeze), JWL Fit]

---

## 三、全量测试报告

### 3.1 爆轰性能对标 (9种代表性炸药)
[Table with columns: 序号, 炸药, 密度, D, P, T, Q]
[Format: Predicted / Experimental]
[Highlight errors: Bold for >10%, Red for >30%]

### 3.2 JWL 参数对标 (完整 5 参数)
[Table with columns: 序号, 炸药, A, B, R1, R2, w]
[Format: Predicted / Experimental (Error%)]
**CRITICAL**: You MUST calculate and display the percentage error for EACH parameter (A, B, R1, R2, w) separately. Do not just list the predicted value.
Example: `778.0 / 750.0 (+3.7%)`
[Highlight errors]

### 3.3 Miller-PDU V3 铝粉反应度输出
[Table of aluminized explosives results]

---

## 四、误差超过 10% 的参数统计

### 4.1 统计汇总
[Summary table of counts]

### 4.2 红色警戒级 (|Error| > 30%)
[List of critical errors with related code module]

### 4.3 黄色警示级 (|Error| 10%-30%)
[List of warning errors]

---

## 五、关键问题分析与咨询课题

[Detailed analysis sections for major issues like: Aluminum Pressure Underestimation, Heat Release Overestimation, JWL Parameter Drift, Liquid Explosive Temperature, etc.]

---

## 六、咨询问题清单

[Summary table of Q1-Q10 ...]

---

## 七、附录

### 7.1 实验参考值来源
[Reference table]

### 7.2 JCZ3 极性参数
[JSON snippet]

---
**文档编制**: Antigravity (PDU Dev Team)
**日期**: YYYY-MM-DD
```
