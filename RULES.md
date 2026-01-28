# PyDetonation-Ultra (PDU) 开发规则

## 0. 项目目标与核心功能

### 0.1 正向计算：全面爆轰性能预测

**输入**：配方（组分 + 比例）

**输出**：
| 类别 | 参数 |
|------|------|
| 基础性能 | 爆速 $D$、爆压 $P_{CJ}$、爆温 $T_{CJ}$、CJ点密度 $\rho_{CJ}$ |
| 能量参数 | 爆热 $Q$、比冲 $I_{sp}$（可选） |
| 产物信息 | 爆轰产物组成、摩尔分数（含气固分离） |
| 氧平衡 | $OB\%$ — 氧平衡百分比 |
| 感度估计 | $h_{50}$ — Kamlet公式撞击感度估计值 (仅供参考) |
| **JWL状态方程参数** | $A$, $B$, $R_1$, $R_2$, $\omega$, $E_0$ — **对数域拟合 + CJ锚定，核心输出** |

### 0.2 逆向设计：多目标约束反推配方

**输入**：目标性能参数组合（可选一种或多种）
- 示例约束：目标爆速 8500 m/s + 目标密度 1.8 g/cm³ + 目标爆压 30 GPa

**输出**：满足约束的配方（组分 + 比例）

### 0.3 技术路线

使用 **JAX 可微分编程** 实现端到端梯度传播，支持：
- 配方参数 → 性能参数的梯度计算
- 基于梯度的配方优化（逆向设计）

---

## 1. 开发环境规范

### 1.1 虚拟环境
- **必须**在 conda 虚拟环境 `nnrf` 下进行所有开发工作
- 激活环境命令：`mamba activate nnrf`

### 1.2 依赖安装
- **首选**：`mamba install -c conda-forge <package_name>`
- **备选**：若 conda-forge 无对应包，使用 `pip install <package_name>`
- **注意**：安装前需确认环境已激活，避免污染系统环境

### 1.3 核心依赖列表
```bash
# JAX GPU 版本 (CUDA 12)
mamba install -c conda-forge jax jaxlib cuda-nvcc

# 科学计算
mamba install -c conda-forge numpy scipy

# 优化器
mamba install -c conda-forge optax

# 数据处理与可视化
mamba install -c conda-forge pandas matplotlib
```

---

## 2. 代码架构原则

### 2.1 混合精度策略 (Mixed Precision)
- **Tier 1 (Float32)**：分子势能计算、矩阵运算、神经网络推理
- **Tier 2 (Float64)**：吉布斯自由能累加、牛顿法残差检查、状态更新
- **原则**：计算密集用 FP32 加速，数值敏感用 FP64 保精度

```python
# 精度转换范式
def compute_intensive_func(x):
    x = x.astype(jnp.float32)  # 下转型加速
    result = heavy_computation(x)
    return result.astype(jnp.float64)  # 上转型保精度
```

### 2.2 隐式微分 (Implicit Differentiation)
- **禁止**：记录牛顿法迭代的完整计算图（会导致 OOM）
- **必须**：使用 `@custom_vjp` 实现隐式微分，只在收敛点计算梯度
- **原则**：前向求解不存梯度，反向传播解伴随方程

### 2.3 对数域变量 (Log-Space)
- **必须**：使用 $z_i = \ln(n_i)$ 替代直接求解摩尔数 $n_i$
- **理由**：避免小量下溢，保证 $n_i > 0$

### 2.4 动态掩码 (Dynamic Masking)
- 根据元素守恒预生成 `active_mask`
- 对不可能生成的产物直接 Mask，减少无效计算

---

## 3. RTX 4060 硬件优化原则

### 3.1 显存管理
```python
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"  # 保留 15% 显存
```

### 3.2 Batch Size 限制
- **推荐**：512（黄金值）
- **上限**：1024（超过会导致 L2 Cache 命中率下降 + OOM 风险）

### 3.3 JIT 编译
- **必须**：所有计算函数使用 `@jax.jit` 装饰器
- **禁止**：在热路径中使用 Python 原生循环
- **推荐**：使用 `jax.lax.while_loop` 替代 Python 循环

### 3.4 调试配置
```python
# 开发阶段开启 NaN 检测
jax.config.update("jax_debug_nans", True)
```

---

## 4. 物理正确性原则

### 4.1 守恒律验证
- 每次迭代后检查质量守恒、元素守恒
- 能量守恒残差目标：< $10^{-6}$

### 4.2 梯度验证
- 使用数值差分 (`finite_difference`) 与 `jax.grad` 对比
- 误差阈值：< $10^{-5}$

### 4.3 相变处理
- **前向**：严谨的 `if-else` 离散逻辑
- **反向**：固定相态假设（Straight-Through Estimator）

---

## 5. 测试与验证规范

### 5.1 单元测试
- 每个核心模块必须有对应的测试用例
- 使用 `pytest` 框架

### 5.2 物理验证
- 对比 EXPLO5 标准数据（RDX、TNT、HMX）
- 误差阈值：爆速 < 2%，CJ压力 < 3%

### 5.3 性能基准
- 单状态点计算时间目标：< 10ms
- 显存占用目标（含梯度）：< 2GB

---

## 6. 代码风格

### 6.1 命名规范
- 函数名：`snake_case`（如 `compute_gibbs_energy`）
- 类名：`PascalCase`（如 `EquilibriumSolver`）
- 常量：`UPPER_SNAKE_CASE`（如 `GAS_CONSTANT`）

### 6.2 文档要求
- 所有公共函数必须有 docstring
- 物理公式必须在注释中标明出处

### 6.3 类型标注
- 使用 Python type hints
- JAX 数组类型：`jax.Array` 或 `jnp.ndarray`

---

## 7. Git 工作流

### 7.1 分支策略
- `main`：稳定版本
- `dev`：开发分支
- `feature/<name>`：功能分支

### 7.2 提交规范
- 格式：`<type>: <description>`
- 类型：`feat`, `fix`, `docs`, `refactor`, `test`, `perf`

---

## 8. 项目结构规划

```
PyDetonation-Ultra/
├── pdu/                    # 核心包
│   ├── __init__.py
│   ├── core/               # 核心求解器
│   │   ├── equilibrium.py  # 平衡求解器 + 隐式微分
│   │   ├── newton.py       # 混合精度牛顿法
│   │   └── cj_search.py    # CJ 点搜索 (Brent 方法)
│   ├── physics/            # 物理模块
│   │   ├── eos.py          # JCZ3 状态方程 (产物EOS)
│   │   ├── jwl.py          # JWL 状态方程拟合 (核心输出!)
│   │   ├── thermo.py       # 热力学函数 (NASA 多项式)
│   │   └── potential.py    # 分子势能函数 (Exp-6)
│   ├── ai/                 # AI 加速模块
│   │   ├── warmstart.py    # Tiny-MLP 初值预测
│   │   └── projection.py   # 原子守恒投影层
│   ├── inverse/            # 逆向设计模块
│   │   ├── optimizer.py    # 多目标优化器 (optax)
│   │   ├── loss.py         # 多目标 Loss 函数
│   │   └── constraints.py  # 配方约束 (元素比例、密度等)
│   └── utils/              # 工具函数
│       ├── precision.py    # 精度控制
│       ├── masking.py      # 动态掩码
│       └── outputs.py      # 输出格式化 (所有性能参数)
├── data/                   # 热力学数据库
│   ├── nasa_polynomials.json
│   └── jcz3_params.json    # JCZ3 势能参数
├── tests/                  # 测试
├── examples/               # 示例脚本
│   ├── forward_demo.py     # 正向计算示例
│   └── inverse_demo.py     # 逆向设计示例
├── docs/                   # 文档
└── RULES.md                # 本文件
```
