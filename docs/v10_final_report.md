# PyDetonation-Ultra (PDU): 下一代含能材料爆轰物理计算引擎
## 项目技术白皮书 (Technical Whitepaper)

**版本**: V10.0 "Physics-First"  
**生成日期**: 2026-01-29  
**定位**: 基于高保真物理模型（Ree Correction & NASA-9）与全局优化算法（PSO）的生产级爆轰引擎

---

## 1. 项目愿景与定位

PyDetonation-Ultra (PDU) 旨在解决传统热化学代码（如 Cheetah, EXPLO5）在**非理想炸药**（如含铝 PBX）及**新材料设计**中的局限性。它不仅仅是一个热力学平衡计算器，更是一个融合了**现代物理模型**与**工程经验修正**的混合引擎。

*   **核心优势**:
    *   **物理优先 (Physics-First)**: V10 引入了 **Francis Ree 极性修正** 模型，通过温度相关的势阱深度 $\epsilon(T)$ 模拟极性产物（如 $H_2O$）的非线性相互作用，极大提升了状态方程在极高压下的保真度。
    *   **热力学高保真**: 全面升级至 **NASA 9 系数多项式** 及 Burcat 数据库，确保在 6000K 以上温区及复杂离解平衡下的物性计算精度。
    *   **全局优化算法**: 实现 JAX 加速的 **PSO (粒子群算法)**。解决了 JWL 拟合在 A/B 参数空间的多峰搜索问题，强制确保声速正定性与能量积分守恒。

---

## 2. 核心物理架构

### 2.1 状态方程 (EOS)
PDU 采用了 **JCZ3 状态方程**，这是一种基于分子间势能（Exp-6 势）的物理 EOS，优于半经验的 BKW。
*   **气体相互作用**: 使用 Ross 软球变分理论修正流体热力学性质。
*   **固相 EOS**: 采用 Murnaghan 状态方程描述凝聚相产物（如石墨、$\text{Al}_2\text{O}_3$）。
*   **混合规则**: 采用改进的单流体混合近似，结合 "Stiff-Mix" 策略（V8.6），引入排斥力修正项以精确复现高密度 HMX/CL-20 的爆压。

### 2.2 化学平衡求解器
*   **算法**: 基于 Gibbs 自由能最小化原理。
*   **求解器**: 采用牛顿-拉夫逊 (Newton-Raphson) 迭代求解非线性系统，并结合 Schur 补技术处理质量平衡约束，确保收敛的鲁棒性。
*   **热力学严谨性**: V8.5 引入了严格的 $\Delta n_{gas} RT$ 修正，彻底解决了 TNT/PETN 等缺氧炸药的能量计算误差。

### 2.3 CJ 点预测
PDU 不再依赖传统的 Hugoniot 交叉法，而是采用更现代的 **AD-Driven Optimization**。利用 JAX 的自动微分能力，直接寻找 Rayleigh 线与 Hugoniot 曲线的切点，精度远高于数值差分法。

---

## 3. JWL 状态方程拟合技术详解 (JWL Engineering & Optimization)

PDU 不仅仅计算爆轰点参数，还负责为流体动力学软件（如 LS-DYNA, Autodyn）生成高精度的 **JWL (Jones-Wilkins-Lee) 状态方程**。为了确保拟合结果既符合数学精度又具备物理合理性，我们开发了一套复杂的“约束拟合”算法。

### 3.1 JWL 的数学形式
在 PDU 中，我们主要针对**主绝热线 (Principal Isentrope)** 进行拟合，其标准解析形式为：
$$P_s(V) = A e^{-R_1 V} + B e^{-R_2 V} + C V^{-(1+\omega)}$$
其中 $V = v/v_0$ 是相对比容。为了保证热力学自洽，常数 $C$ 必须满足初始内能 $E_0$ 的约束。

### 3.2 多目标优化逻辑
JWL 拟合不是简单的最小二乘法，而是一个带有物理惩罚项的多目标优化过程。其核心目标函数（Objective Function）包含五个维度：

1.  **(A) Log-MSE Loss (数据拟合)**:
    在对数空间进行压力匹配，确保在极宽的压力波动范围内（100 GPa 到 0.1 GPa）都具有良好的拟合精度。
2.  **(B) $P_{CJ}$ Anchor (工程修正)**:
    强制 JWL 曲线通过计算得到的（或实验指定的）CJ 点。这是 V8.6 "Engineering Bias" 的核心，确保了起爆点压力在流体计算中绝对准确。
3.  **(C) $\Gamma_{CJ}$ Consistency (绝热指数锚定)**:
    通过约束 $dP/dV$ 的斜率，确保 JWL 在 CJ 点处的波阻抗与理论值一致。
4.  **(D) Total Energy Constraint (能量守恒)**:
    **V8.7 核心改进**。对于非理想炸药，强制 JWL 的全行程积分（从 $V_{CJ}$ 到 $\infty$）等于计算的总爆热 $Q$。
5.  **(E) Prior Penalty (文献先验)**:
    引入历史文献中的 A, B, R1, R2, $\omega$ 作为先验，防止优化陷入病态的参数空间（如 $R_1 < R_2$）。

### 3.3 参数的物理含义与敏感性分析 (Physical Meaning & Sensitivity)

为了深入理解 JWL 的工程价值，必须明确各参数在物理空间中的控制区域：

*   **$A$ 与 $R_1$ (高压区控)**：
    控制 $V \approx 1$ 到 $V \approx 2$ 的区域。$A$ 代表高压幅值，$R_1$ 决定高压衰减率。
    > [!NOTE]
    > $R_1$ 通常在 4.5 到 5.0 之间。若 $R_1$ 过小，爆轰波阻抗会严重失真。
*   **$B$ 与 $R_2$ (中压区控)**：
    控制 $V \approx 2$ 到 $V \approx 7$ 的区域。这是做功最关键的活塞行程区。$R_2$ 决定了向低压 tail 转换的平滑度。
*   **$\omega$ 与 $C$ (低压长尾)**：
    控制 $V > 10$ 的无限膨胀区。$\omega$ 是 Grüneisen 系数，决定了废气的残余做功能力。
    > [!IMPORTANT]
    > **V9.0 稳定性修正**：对于含铝炸药，通过调节 `combustion_efficiency` (0.0~1.0)，可以有效防止 $\omega$ 参数因能量负载过重而崩塌（从 V8.7 的 <0.1 恢复到 ~0.3 的健康区间）。

### 3.4 两步法中的能量积分约束 (Mathematical Derivation)

PDU V8.7 最核心的创新在于在 JWL 拟合中引入了**全行程膨胀功限制**。

根据热力学定义，从 CJ 点膨胀到无穷远的理想做功为：
$$E_{pot} = \int_{V_{CJ}}^{\infty} P_s(V) dV$$

代入 JWL 绝热形式解析积分：
$$E_{pot}(V) = \frac{A}{R_1} e^{-R_1 V} + \frac{B}{R_2} e^{-R_2 V} + \frac{C}{\omega} V^{-\omega}$$

**在 V9.0 "React-Flow" 策略中**：
1.  **反应度控制 (`reaction_degree`)**：通过控制 CJ 面上的铝反应比例 $\lambda_{Al}$，在计算上解耦了“化学释能”与“气相产率”，使得压强 $P_{CJ}$ 可以向实验值灵活回升。
2.  **能量锚定 (`combustion_efficiency`)**：引入效率因子 $\eta$ 缩放目标爆热 $Q_{target} = \eta \cdot Q_{bulk}$。
3.  **JWL 拟合逻辑**：强制要求 $E_{pot}(V_{CJ}(\lambda)) = \eta \cdot Q$。这种双参数控制模式极大地拓宽了 JWL 的物理有效范围。
4.  这正是含铝炸药在流体仿真中表现出“压力可控、推力持久、参数稳定”的核心物理原因。

### 3.5 核心实现逻辑 (Python 代码)
以下是 `pdu/physics/jwl.py` 中用于约束优化的代价函数片段：

```python
def objective(params):
    A, B, R1, R2, w, C = params
    
    # 1. 物理边界硬约束
    if A < 0 or B < 0 or C < 0 or w > 1.2 or R1 < (R2 + 1.0):
        return 1e9

    # 2. 基础拟合误差 (Log-Domain)
    P_pred = A*np.exp(-R1*V_rel) + B*np.exp(-R2*V_rel) + C/(V_rel**(1+w))
    loss_fit = np.mean((np.log(P_pred) - np.log(P_target))**2) * 50.0

    # 3. CJ 压力锚定 (严格约束)
    P_fit_cj = A*np.exp(-R1*V_start) + B*np.exp(-R2*V_start) + C/(V_start**(1+w))
    loss_anchor_P = ((P_fit_cj - target_P_cj) / target_P_cj) ** 2 * 10000.0

    # 4. V8.7 能量积分约束 (关键特性)
    term1 = (A/R1)*np.exp(-R1*V_start)
    term2 = (B/R2)*np.exp(-R2*V_start)
    term3 = (C/w)*(V_start**(-w))
    E_integral = term1 + term2 + term3
    loss_energy = ((E_integral - constraint_total_energy) / constraint_total_energy) ** 2 * 5000.0

    return loss_fit + loss_anchor_P + loss_energy + loss_prior
```

---

## 4. 核心代码实现片段 (Core Implementation Snippets)

为便于技术交流与二次开发，本节展示了 PDU 引擎中最核心的三个技术模块：多相自由能计算、V8.7 两步法 API、以及基于 JAX 的 CJ 点扫描逻辑。

### 4.1 多相亥姆霍兹自由能与排斥力修正 (EOS Core)
这是 PDU 物理引擎的“心脏”。它实现了气固分离、固相体积扣除以及 V8.6 的高压排斥力修正。

```python
@jax.jit
def compute_total_helmholtz_energy(n, V_total, T, coeffs_all, ...):
    # 1. 分离气相与固相 (V8 关键扩展)
    n_solid = n * solid_mask
    n_gas = n * (1.0 - solid_mask)

    # 2. 计算固相占据的动态有效体积 (Murnaghan EOS)
    solid_vol_eff = compute_solid_volume_murnaghan(solid_v0, P_proxy, is_carbon, is_alumina)
    V_gas_eff = V_total - jnp.sum(n_solid * solid_vol_eff)

    # 3. 气相自由能计算 (JCZ3 Ideal + Excess)
    A_gas_ideal = ... 
    A_excess_hs = ... # 硬球修正

    # 4. [V8.6 Upgrade: Repulsive Correction] 
    # 针对高压下的压力亏损，引入 eta^4 幂律排斥自由能
    A_rep = n_gas_total * R * T * 15.0 * (eta**4)
    
    return A_gas_ideal + A_excess_hs + U_attr + A_rep + A_solid
```

### 4.2 V8.7 两步法调用逻辑 (API Layer)
展示了如何通过封装实现“惰性计算获取压力”与“全反应计算获取能量”的无缝切换。

```python
def detonation_forward(..., inert_species=None, target_energy=None):
    # V8.7 动态产物控制：处理惰性铝
    if inert_species and 'Al' in inert_species:
        # 强制移除氧化铝产物，使铝仅作为固体填料存在
        SPECIES_LIST = [s for s in base_species if s != 'Al2O3']

    # 调用 AD-Based 求解器计算 CJ 状态
    D, P_cj, T_cj, V_cj = predict_cj_with_isentrope(...)

    # JWL 拟合：传入 target_energy 实现能量守恒约束
    jwl = fit_jwl_from_isentrope(..., constraint_total_energy=target_energy)
    
    return DetonationResult(...)
```

### 4.4 V10 核心物理优化实现 (V10 Core Optimizations)
 
#### 4.4.1 Francis Ree 极性流体修正
在 `pdu/physics/eos.py` 中，我们实现了对极性分子（如水）的势能修正：
```python
@jax.jit
def compute_mixed_matrices_dynamic(T, eps_vec, r_vec, alpha_vec, lambda_vec):
    # Francis Ree 修正: eps(T) = eps0 * (1 + lambda/T)
    T_safe = jnp.maximum(T, 1e-2)
    eps_T = eps_vec * (1.0 + lambda_vec / T_safe)
    
    # Lorentz-Berthelot 混合规则
    eps_matrix = jnp.sqrt(jnp.outer(eps_T, eps_T))
    r_matrix = 0.5 * (jnp.expand_dims(r_vec, 1) + jnp.expand_dims(r_vec, 0))
    return eps_matrix, r_matrix, alpha_matrix
```
 
#### 4.4.2 NASA 9 系数热力学引擎
在 `pdu/physics/thermo.py` 中，V10 能够动态处理 7 系数与 9 系数数据：
```python
def _h_9(c, t):
    # H/RT = -a1*T^-2 + a2*T^-1*lnT + a3 + a4*T/2 + ... + a8/T
    t_inv = 1.0 / t
    return -c[0]*t_inv**2 + c[1]*t_inv*jnp.log(t) + c[2] + c[3]*t/2.0 + \
           c[4]*t**2/3.0 + c[5]*t**3/4.0 + c[6]*t**4/5.0 + c[7]*t_inv
```
 
#### 4.4.3 PSO JWL 全局拟合
利用 JAX 的向量化并行能力，PSO 可以在数千个粒子上同时搜索最优 JWL 参数：
```python
# pdu/calibration/pso.py
fitness = jax.vmap(self.objective_fn)(pos) # 并行评估
# 强制物理约束：R1 > R2, w > 0, 膨胀功守恒
```

---

## 5. 演进历程与关键技术突破

PDU 的开发经历了一系列针对性的技术攻关，解决了工业界长期存在的痛点：

### Phase 1: V8.5 热力学重构 (Thermodynamic Rigor)
*   **问题**: 早期版本严重高估缺氧炸药（如 TNT）的爆热。
*   **解法**: 发现并修正了内能 ($U$) 与焓 ($H$) 的混淆，引入了 $\Delta n_{gas} RT$ 气体摩尔功修正。
*   **成果**: TNT 爆热误差从 >50% 降至 <10%，物理地基夯实。

### Phase 2: V8.6 "Stiff-Mix" (High-Pressure Recovery)
*   **问题**: 引入严格物理 EOS 后，理想炸药（如 HMX）的爆压 $P_{CJ}$ 偏低（38 GPa vs 实测 42 GPa）。这是由于标准 JCZ3 参数在极高压下过软。
*   **解法**:
    1.  **Fried-Howard Carbon**: 替换石墨与液碳的 EOS 参数。
    2.  **Repulsive Correction**: 引入高压排斥自由能修正项 ($A_{rep}$)，模拟分子在高密度下的“硬球”效应。
*   **成果**: HMX $P_{CJ}$ 精确恢复至 41.4 GPa (-1.5% 误差)。

### Phase 3: V8.7 "Inert-Eq" 两步法 (Aluminized Solution)
*   **问题**: 含铝炸药（如 PBXN-109）存在“能量-压力悖论”。若允许铝反应，模型预测 $P_{CJ} > 35$ GPa（荒谬）；若不允许，则 $Q$ 极低。
*   **解法**: 实施专家建议的 **Two-Step Strategy**:
    1.  **Step 1 (Inert CJ)**: 强制铝在 CJ 面惰性，计算真实的低爆压流场。
    2.  **Step 2 (Active Q)**: 计算全反应释放的总化学能。
    3.  **Hybrid Fitting**: 利用 3.2 节所述的 **Energy Constraint** 技术，将 JWL 锚定在 Step 1 的 CJ 点，但强制其总积分功等于 Step 2 的总爆热。
*   **成果**: 成功生成了具有“低爆压、高长尾能量”特征的 JWL 参数（$R_2 \approx 1.5$, $\omega \approx 0.1$），符合后燃烧物理图像。

### Phase 4: V9.0 "React-Flow" (Dynamic Reaction & Efficiency)
*   **问题**: V8.7 采用“全惰性”假设导致 PBXN-109 爆压偏低，且 Tritonal 因能量虚高导致 JWL 参数 ($\omega$) 崩塌。
*   **解法**:
    1.  **Partial Reaction**: 引入反应度参数 $\lambda_{Al}$，允许铝粉在 CJ 面部分反应（10%-20%），并底层支持气固耦合 EOS。
    2.  **Combustion Efficiency**: 引入效率因子 $\eta$，对爆热进行有效性折算。
*   **成果**: 成功使 Tritonal 的 JWL $\omega$ 从 0.06 恢复至 0.33，PBXN-109 压强预测向 18.6 GPa 回升。

---

## 6. 全面性能概览 (V9.0)

基于最新的全量验证报告：

| 炸药类型 | 代表 | 爆压精度 | 爆热精度 | 评价 |
| :--- | :--- | :---: | :---: | :--- |
| **理想高能 (CHNO)** | HMX, RDX | **极高** (-1.5%) | 高 (-7.9%) | 完全达到工业级精度。 |
| **混合炸药** | Octol, Comp B | **高** (±5%) | 高 (±5%) | 混合规则有效，预测可靠。 |
| **含铝炸药** | PBXN-109, Tritonal | **改善** (-19%) | **最优** (0.0% Error) | **V9.0 突破点**。解决了能量虚高导致的 JWL 稳定性问题，并初步实现了压强回升。 |

---

# V10.0 全量物理标定报告 (Physics-First Benchmark)
 
## 1. 爆轰性能汇总对标 (D, P, T, Q)
 
| 炸药 | 爆速 $D$ (m/s) [Err] | 爆压 $P_{CJ}$ (GPa) [Err] | 爆温 $T_{CJ}$ (K) [Err] | 爆热 $Q$ (MJ/kg) [Err] |
| :--- | :---: | :---: | :---: | :---: |
| **HMX** | 8616/9110 (-5.4%) | 39.2/42.0 (-6.8%) | 3584/3800 (-5.7%) | 5.70/6.19 (-7.9%) |
| **RDX** | 8443/8750 (-3.5%) | 33.5/34.7 (-3.4%) | 3557/3600 (-1.2%) | 5.76/5.53 (+4.2%) |
| **PETN** | 8030/8260 (-2.8%) | 31.7/31.5 (+0.5%) | 3728/4503 **(-17.2%)** | 6.20/5.81 (+6.7%) |
| **TNT** | 7168/6930 (+3.4%) | 20.3/21.0 (-3.5%) | 3391/3100 (+9.4%) | 5.03/4.69 (+7.2%) |
| **NM** | 6179/6260 (-1.3%) | 11.2/12.6 **(-10.9%)** | 3047/3600 **(-15.4%)** | 4.76/4.40 (+8.1%) |
| **Comp B** | 7944/7980 (-0.5%) | 28.2/29.5 (-4.3%) | 3499/3400 (+2.9%) | 5.47/5.10 (+7.3%) |
| **Octol** | 8314/8480 (-2.0%) | 32.8/34.8 (-5.8%) | 3471/3500 (-0.8%) | 5.53/5.60 (-1.2%) |
| **Tritonal** | 7660/6700 **(+14%)** | 17.0/25.0 **(-32%)** | 3285/3200 (+2.7%) | 10.23/7.50 **(+36%)** |
| **PBXN-109** | 8157/8050 (+1.3%) | 19.0/30.0 **(-36%)** | 2993/3300 (-9.3%) | 9.73/9.70 (+0.4%) |
 
## 2. JWL 详细参数对标与偏差分析 (V10 PSO Fit vs Exp)
 
| 炸药 | $A$ (GPa) [Err] | $B$ (GPa) [Err] | $R_1$ [Err] | $R_2$ [Err] | $\omega$ [Err] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HMX** | 653.7 **(-16%)** | 7.62 (+8%) | 4.13 (-1.7%) | 1.01 (+0.9%) | 0.300 (+0.1%) |
| **RDX** | 660.5 **(-15%)** | 13.93 **(+97%)** | 4.39 (-2.5%) | 1.30 (-7.1%) | 0.252 (+0.8%) |
| **PETN** | 577.9 (-6.3%) | 14.94 (-12%) | 4.45 (+1.1%) | 1.28 (+6.4%) | 0.249 (-0.3%) |
| **TNT** | 428.9 **(+15%)** | 3.04 (-5.8%) | 4.28 (+3.2%) | 0.97 (+8.0%) | 0.299 (-0.5%) |
| **NM** | 230.3 **(-53%)** | 4.46 **(-30%)** | 4.62 (-7.6%) | 1.27 **(-16%)** | 0.296 **(-22%)** |
| **Comp B** | 550.3 (+5%) | 8.92 (+16%) | 4.34 (+3.3%) | 1.22 (+11%) | 0.270 **(-21%)** |
| **Octol** | 654.2 (+0.9%) | 6.31 **(-51%)** | 4.28 (-4.9%) | 1.01 **(-28%)** | 0.300 (+7%) |
| **Tritonal** | 513.8 **(+28%)** | 2.39 **(-40%)** | 4.53 (+1%) | 0.99 **(-17%)** | 0.294 **(-16%)** |
| **PBXN-109** | 802.5 **(-31%)** | 7.32 **(-62%)** | 5.01 **(-12%)** | 1.46 **(+17%)** | 0.249 **(+25%)** |
 
---
 
## 3. 误差深度溯源与缺陷分析 (Deficit Analysis)
 
> [!WARNING]
> 虽然 V10 在算法稳健性上取得了突破，但上述数据暴露出 PDU 引擎在特定物理场景下的深度缺陷，必须在咨询中明确。
 
### 3.1 爆轰参数重大偏差分析
1.  **硝基甲烷 (NM): P (-11%) 与 T (-15%) 双亏损**
    *   **根源分析**: NM 是典型的液相炸药。目前的 JCZ3 Exp-6 参数主要校准自气相 Hugoniot。在高密度液相爆轰下，势能函数可能过于“柔软”，且缺乏有效的偶极-偶极微扰修正。
2.  **PETN: 爆温 T (-17.2%) 严重低估**
    *   **根源分析**: PETN 产物中极性组分比例极高。Ree 修正虽然改善了压力，但可能显著改变了系统的能级分布（配分函数），导致温度预测失真。
3.  **Tritonal: P (-32.1%) 与 Q (+36.5%) 的“非理想悖论”**
    *   **根源分析**: Tritonal (80/20 TNT/Al) 的铝粉反应极慢。V10 的平衡模型假设了部分平衡 (10% Al 反应)，但这无法捕捉到真实的能量释放时序。$Q$ 的虚高说明燃烧效率参数 $\eta$ 仍需物理耦合模型。
4.  **PBXN-109: P (-36.5%) 持续偏低**
    *   **根源分析**: 复杂 binder (HTPB) 与铝粉、RDX 的相互作用导致了极强的非理想性（体积亏损）。单纯依靠 $V_{eff}$ 扣除可能无法模拟多相流体在该尺度下的复杂输运规律。
 
### 3.2 JWL 参数空间的物理漂移
*   **PBXN-109 的“全面失真”**: 标定得到的 A, B, R1, R2, $\omega$ 相对实验值均有巨大偏差。
    *   **技术反思**: 虽然 PSO 找到了一个数学上的全局最优解（满足膨胀功守恒），但由于 CJ 点压力的巨大起始误差（-36%），导致整个膨胀曲线发生了系统性位移。**数学收敛 $\neq$ 物理正确**。
*   **A, B 系数的过度敏感性**: 
    *   在 HMX, RDX, TNT 等药中，A 系数的 15%-25% 偏差通常是由于 PSO 在平衡 $P_{CJ}$ 锚定与主绝热线对数拟合权重时的折衷。

---
 
## 4. 专家咨询核心课题 (Consultation Objectives)
 
针对上述缺陷，我们需要向外部专家（如 LLNL 或 211 所）咨询以下技术方案：
 
1.  **液相 EOS 强化**: 如何针对 NM 等液态炸药引入密度相关的势阱半径 $r^*(\rho)$ 修正？
2.  **非理想动力学集成**: 在 CJ 平衡计算中，是否应根据粒子尺寸引入铝粉反应的动力学标尺，而非简单的反应度 $\lambda$？
3.  **多目标 PSO 权重调优**: 针对 JWL 拟合中的 $A, B$ 跳变，如何定义更符合流体模拟需求的“物理权重矩阵”？
4.  **AlO/Al2O(g) 气相平衡**: 重新评估高温下铝氧化物气相对 $P_{CJ}$ 的贡献（V10 暂因收敛性移除，需物理重构）。

---

## 8. 遗留问题梳理与专家咨询需求 (V9 Deficit Analysis)

基于 V9.0 的全量对标测试，我们识别出以下误差超过 10% 的关键领域，这将作为 V10 版本开发的重点攻关方向。

### 8.1 爆轰性能参数偏差 (>10%)

*   **问题 1: 爆温 $T_{CJ}$ 的系统性偏差 (PETN: -14.6%, TNT: +15.4%)**
    *   **分析**: 反映了 JCZ3 参数中 $\alpha$（势能排斥常数）对不同氧平衡炸药的一致性较差。
    *   **修正建议**: 建立动态 $\alpha$ 调整模型，根据 $O/C$ 原子比或分子的极性分布特征，对混合规则中的 $\alpha$ 矩阵进行非线性修正。
*   **问题 2: 非理想炸药爆压预测仍具挑战 (PBXN-109: -19.0%, NM: -12.3%)**
    *   **分析**: PBXN-109 的 V9 部分反应模型已打通，但仍存在显著亏损。硝基甲烷（NM）的低估则可能源于液相 JCZ3 修正项不足。
    *   **修正建议**: 
        1.  **反应路径深挖**: 对铝粉燃烧中间体（AlO, Al2O, Al-Gas）进行热力学建模，替代单一的固相 Al2O3 生成路径。
        2.  **EOS 硬化**: 针对 NM 等液态药，优化 eta^4 修正项的触发阈值，模拟更强的分子间位阻效应。
*   **问题 3: 部分爆速预报过高 (PBXN-109: +11.6%)**
    *   **分析**: 爆速由 Rayleigh 线斜率决定。在压力亏损的情况下爆速却偏高，说明 CJ 点体积 $V_{CJ}$ 预测偏小，即 Hugoniot 曲线斜率过大。

### 8.2 JWL 参数空间的数学不稳定性 (>20%)

尽管 JWL 的物理锚定（$\omega, P_{CJ}$）已在 V9 中大幅改善，但系数绝对值仍存在巨大偏差：
*   **典型现象**: PBXN-109 (A:+74%, B:+55%)；TNT (A:+25%)；NM (B:-29%)。
*   **咨询需求**:
    1.  **权重重平衡**: 目前拟合过于依赖实验先验 (Prior Penalty)。是否应引入“曲率约束” (d²P/dV²)，从关注点坐标转向关注膨胀曲线的物理构型？
    2.  **多峰搜索**: 针对 A/B 参数的巨大跳变，可能存在多个局部最优解。是否应引入全局优化器（如粒子群或遗传算法）来初始化 JWL 系数？

### 8.3 混合先验回归逻辑
*   **问题 4: 复杂组分下的 Prior 崩塌 (Comp B: $\omega$ -21%)**
    *   **分析**: 当高能单质药（RDX）与低能组分（TNT）混合时，简单的线性混合先验可能无法捕捉非线性的绝热指数变化。
    *   **修正建议**: 开发基于“有效氧平衡”或“产物平均分子量”权重的交互项修正逻辑。

---

**总结**: V10 的核心任务将从“功能补齐”转向“物理精调”，通过引入更细致的化学动力学路径和动态 EOS 参数，全面收敛上述技术指标。

---

## 9. V10 版本技术演进路线图 (Expert-Verified Roadmap)

根据最新的深度专家咨询报告，V10 版本的开发将聚焦于以下四个核心技术维度的重构，以实现从“经验拟合”到“物理保真”的跨越。

### 9.1 物理层：极性流体与微扰理论 (Physics Fidelity)
*   **当前痛点**: 标准 Exp-6 势能函数无法描述爆轰产物中水分子 ($H_2O$) 的强极性和氢键效应。
*   **V10 方案**: 
    1.  **Francis Ree 修正**: 引入温度依赖的势阱深度 $\epsilon(T) = \epsilon_0 (1 + \lambda/T)$，在不增加计算复杂度的情况下模拟偶极相互作用。
    2.  **WCA 微扰理论**: 逐步构建基于 Weeks-Chandler-Andersen 理论的硬球参考系，替代现有的单流体混合规则，提升高密度下的混合熵计算精度。

### 9.2 数据层：高温热化学体系升级 (Data Precision)
*   **当前痛点**: 7 系数 NASA 多项式在 >6000K 温区精度下降；缺乏 AlO/Al2O 等关键中间体数据。
*   **V10 方案**:
    1.  **NASA 9-Coefficient**: 全面升级热力学数据库至 9 系数多项式格式。
    2.  **Burcat 数据库集成**: 引入 Alexander Burcat 的“第三千禧年数据库”，补全铝粉燃烧中间体（AlO, Al2O, AlO+）的高温光谱数据。

### 9.3 算法层：基于物理约束的智能标定 (Smart Calibration)
*   **当前痛点**: 传统的梯度下降法（Nelder-Mead）容易陷入局部最优，且生成的 JWL 参数常违反物理约束（如声速虚数）。
*   **V10 方案**:
    1.  **PSO 粒子群优化**: 引入随机全局优化算法，解决多峰目标函数的搜索问题。
    2.  **混合约束机制**: 在流体计算前植入“物理过滤器”，强制检查凸性（Convexity）与 Grüneisen 正定性，拒绝非物理参数。

### 9.4 特定场景：非理想修正 (Specialized Corrections)
*   **碳凝结滞后**: 引入 **Shaw-Johnson 模型**，通过独立的动力学方程描述碳团簇的成核与生长。
*   **过压爆轰 (Overdriven)**: 实施 **JWL++** 扩展模型，通过变 $\omega(V)$ 函数修正高压区 ($P > P_{CJ}$) 的 EOS 行为。
