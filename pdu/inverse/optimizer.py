"""
逆向优化器模块

给定目标性能参数，优化配方组成和密度。
使用梯度下降优化，支持多目标和约束。
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, List, Optional, NamedTuple, Tuple, Callable
from dataclasses import dataclass
from functools import partial


class OptimizationResult(NamedTuple):
    """优化结果"""
    recipe: Dict[str, float]        # 最优配方 (组分: 质量分数)
    density: float                  # 最优密度 (g/cm³)
    performance: Dict[str, float]   # 预测性能
    loss: float                     # 最终损失值
    converged: bool                 # 是否收敛
    n_iters: int                    # 迭代次数
    loss_history: List[float]       # 损失历史


@dataclass
class RecipeConstraints:
    """配方约束"""
    # 组分约束
    min_fractions: Dict[str, float] = None  # 最小质量分数
    max_fractions: Dict[str, float] = None  # 最大质量分数
    required_components: List[str] = None   # 必须包含的组分
    forbidden_components: List[str] = None  # 禁止的组分
    
    # 密度约束
    min_density: float = 1.0                # 最小密度
    max_density: float = 2.2                # 最大密度
    
    # 其他约束
    max_oxygen_balance: float = None        # 氧平衡上限
    min_oxygen_balance: float = None        # 氧平衡下限
    max_sensitivity: float = None           # 感度上限 (h50 下限)
    
    def __post_init__(self):
        if self.min_fractions is None:
            self.min_fractions = {}
        if self.max_fractions is None:
            self.max_fractions = {}
        if self.required_components is None:
            self.required_components = []
        if self.forbidden_components is None:
            self.forbidden_components = []


def multi_objective_loss(
    performance: Dict[str, float],
    targets: Dict[str, float],
    weights: Dict[str, float] = None
) -> float:
    """多目标损失函数
    
    L = Σ w_i * ((y_i - target_i) / scale_i)²
    
    Args:
        performance: 预测性能字典
        targets: 目标性能字典
        weights: 权重字典 (可选)
        
    Returns:
        加权损失值
    """
    if weights is None:
        weights = {k: 1.0 for k in targets.keys()}
    
    # 各参数的归一化尺度
    scales = {
        'D': 1000.0,      # m/s
        'P_cj': 10.0,     # GPa
        'T_cj': 500.0,    # K
        'Q': 500.0,       # kJ/kg
        'rho_cj': 0.5,    # g/cm³
        'OB': 20.0,       # %
    }
    
    loss = 0.0
    for key, target in targets.items():
        if key in performance:
            pred = performance[key]
            scale = scales.get(key, 1.0)
            w = weights.get(key, 1.0)
            loss += w * ((pred - target) / scale) ** 2
    
    return loss


def apply_constraints(
    logits: jnp.ndarray,
    density: float,
    component_names: List[str],
    constraints: RecipeConstraints
) -> Tuple[jnp.ndarray, float, float]:
    """应用约束并计算惩罚
    
    Args:
        logits: 组分 logits
        density: 密度
        component_names: 组分名称列表
        constraints: 约束对象
        
    Returns:
        (修正后的 logits, 修正后的密度, 惩罚项)
    """
    penalty = 0.0
    
    # 密度约束
    density = jnp.clip(density, constraints.min_density, constraints.max_density)
    
    # 禁止组分惩罚
    fracs = jax.nn.softmax(logits)
    for i, name in enumerate(component_names):
        if name in constraints.forbidden_components:
            penalty += 100.0 * fracs[i] ** 2
    
    # 最小/最大分数约束
    for i, name in enumerate(component_names):
        if name in constraints.min_fractions:
            min_frac = constraints.min_fractions[name]
            violation = jnp.maximum(min_frac - fracs[i], 0)
            penalty += 10.0 * violation ** 2
        
        if name in constraints.max_fractions:
            max_frac = constraints.max_fractions[name]
            violation = jnp.maximum(fracs[i] - max_frac, 0)
            penalty += 10.0 * violation ** 2
    
    return logits, density, penalty


def optimize_recipe(
    targets: Dict[str, float],
    available_components: List[str],
    constraints: RecipeConstraints = None,
    weights: Dict[str, float] = None,
    n_iters: int = 500,
    lr: float = 0.05,
    verbose: bool = False,
    seed: int = 42
) -> OptimizationResult:
    """优化配方
    
    Args:
        targets: 目标性能字典，如 {'D': 8500, 'P_cj': 32}
        available_components: 可用组分列表
        constraints: 配方约束
        weights: 目标权重
        n_iters: 迭代次数
        lr: 学习率
        verbose: 是否打印进度
        seed: 随机种子
        
    Returns:
        OptimizationResult
    """
    import optax
    from pdu.data.components import get_component
    
    if constraints is None:
        constraints = RecipeConstraints()
    
    n_comp = len(available_components)
    key = random.PRNGKey(seed)
    
    # 初始化参数
    key, subkey = random.split(key)
    logits_init = random.normal(subkey, (n_comp,)) * 0.1
    density_init = (constraints.min_density + constraints.max_density) / 2
    
    params = {
        'logits': logits_init,
        'density': jnp.array([density_init])
    }
    
    # 性能计算函数（简化版，使用 K-J 公式）
    # 预加载组分数据
    comp_data = []
    for name in available_components:
        c = get_component(name)
        comp_data.append({
            'ob': float(c.oxygen_balance),
            'hof': float(c.heat_of_formation),
            'mw': float(c.molecular_weight),
            'C': float(c.formula.get('C', 0)),
            'H': float(c.formula.get('H', 0)),
            'N': float(c.formula.get('N', 0)),
            'O': float(c.formula.get('O', 0)),
            'Al': float(c.formula.get('Al', 0)),
        })
    
    # 转为 JAX 数组
    ob_arr = jnp.array([d['ob'] for d in comp_data])
    hof_arr = jnp.array([d['hof'] for d in comp_data])
    mw_arr = jnp.array([d['mw'] for d in comp_data])
    C_arr = jnp.array([d['C'] for d in comp_data])
    H_arr = jnp.array([d['H'] for d in comp_data])
    N_arr = jnp.array([d['N'] for d in comp_data])
    O_arr = jnp.array([d['O'] for d in comp_data])
    Al_arr = jnp.array([d['Al'] for d in comp_data])
    
    def compute_performance_jax(fractions, density):
        """纯 JAX 性能计算"""
        # 加权平均
        weighted_ob = jnp.sum(fractions * ob_arr)
        weighted_hof = jnp.sum(fractions * hof_arr)
        equiv_mw = jnp.sum(fractions * mw_arr)
        
        # 等效分子式 (每 100g)
        n_mol = fractions * 100.0 / mw_arr  # mol per 100g
        equiv_C = jnp.sum(n_mol * C_arr)
        equiv_H = jnp.sum(n_mol * H_arr)
        equiv_N = jnp.sum(n_mol * N_arr)
        equiv_O = jnp.sum(n_mol * O_arr)
        equiv_Al = jnp.sum(n_mol * Al_arr)
        
        # 归一化
        scale = jnp.maximum(equiv_mw, 1.0) / 100.0
        
        # 内联 K-J 公式（避免调用外部函数需要 int 转换）
        n_C = equiv_C / scale
        n_H = equiv_H / scale
        n_N = equiv_N / scale
        n_O = equiv_O / scale
        
        # 产物组成估计
        n_H2O = n_H / 2.0
        O_remaining = n_O - n_H2O
        
        n_CO2 = jnp.minimum(n_C, jnp.maximum(O_remaining / 2, 0))
        n_CO = jnp.clip(O_remaining - n_CO2 * 2, 0, n_C - n_CO2)
        n_N2 = n_N / 2.0
        
        # 气体总量和平均分子量
        n_gas = n_H2O + n_CO2 + n_CO + n_N2
        M_gas = jnp.where(n_gas > 0,
            (18.015 * n_H2O + 44.009 * n_CO2 + 28.010 * n_CO + 28.014 * n_N2) / n_gas,
            28.0)
        
        # N: mol/g
        N = n_gas / jnp.maximum(equiv_mw, 1.0)
        
        # 爆热 (cal/g)
        Q_prod = n_H2O * 57.8 + n_CO2 * 94.1 + n_CO * 26.4  # kcal/mol
        Q_exp = weighted_hof / jnp.maximum(equiv_mw, 1.0) / 4.184 * 1000  # kJ/mol -> kcal/mol -> kcal/g
        Q = jnp.maximum((Q_prod / jnp.maximum(scale, 0.01) - Q_exp) * 10, 500.0)  # cal/g
        
        # K-J 计算
        phi = N * jnp.sqrt(M_gas) * jnp.sqrt(Q)
        D = 1.01 * jnp.sqrt(phi) * (1 + 1.30 * density) * 1000  # m/s
        P_cj = 15.58 * density ** 2 * phi  # GPa
        
        # 限制范围
        D = jnp.clip(D, 4000, 10000)
        P_cj = jnp.clip(P_cj, 10, 50)
        
        return D, P_cj, Q, weighted_ob
    
    # 损失函数
    def loss_fn(params):
        fracs = jax.nn.softmax(params['logits'])
        density = jnp.clip(
            params['density'][0],
            constraints.min_density,
            constraints.max_density
        )
        
        # 计算性能
        D, P_cj, Q, OB = compute_performance_jax(fracs, density)
        
        # 构建性能字典
        perf = {'D': D, 'P_cj': P_cj, 'Q': Q, 'OB': OB}
        
        # 多目标损失
        loss = 0.0
        for key, target in targets.items():
            if key in perf:
                pred = perf[key]
                scale = {'D': 1000.0, 'P_cj': 10.0, 'Q': 500.0, 'OB': 20.0}.get(key, 1.0)
                w = weights.get(key, 1.0) if weights else 1.0
                loss = loss + w * ((pred - target) / scale) ** 2
        
        # 约束惩罚
        _, _, penalty = apply_constraints(
            params['logits'], density, available_components, constraints
        )
        
        return loss + penalty
    
    # 优化器
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)
    
    # 优化循环
    loss_history = []
    
    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    for i in range(n_iters):
        params, opt_state, loss = step(params, opt_state)
        loss_history.append(float(loss))
        
        if verbose and i % 100 == 0:
            print(f"Iter {i}: loss = {loss:.4f}")
    
    # 提取最终结果
    final_fracs = jax.nn.softmax(params['logits'])
    final_density = float(jnp.clip(
        params['density'][0],
        constraints.min_density,
        constraints.max_density
    ))
    
    # 构建配方（过滤掉小于 1% 的组分）
    recipe = {}
    for name, frac in zip(available_components, final_fracs):
        if frac > 0.01:
            recipe[name] = float(frac)
    
    # 归一化
    total = sum(recipe.values())
    recipe = {k: v / total for k, v in recipe.items()}
    
    # 最终性能
    D, P_cj, Q, OB = compute_performance_jax(final_fracs, final_density)
    final_perf = {'D': float(D), 'P_cj': float(P_cj), 'Q': float(Q), 'OB': float(OB)}
    
    # 判断收敛
    converged = loss_history[-1] < 1.0 if loss_history else False
    
    return OptimizationResult(
        recipe=recipe,
        density=final_density,
        performance=final_perf,
        loss=loss_history[-1] if loss_history else float('inf'),
        converged=converged,
        n_iters=n_iters,
        loss_history=loss_history
    )


def grid_search_recipe(
    targets: Dict[str, float],
    available_components: List[str],
    density_range: Tuple[float, float] = (1.5, 2.0),
    n_density_points: int = 5,
    n_composition_samples: int = 100,
    seed: int = 42
) -> OptimizationResult:
    """网格搜索最优配方
    
    适用于组分数量较少的情况。
    
    Args:
        targets: 目标性能
        available_components: 可用组分
        density_range: 密度范围
        n_density_points: 密度采样点数
        n_composition_samples: 组成采样数
        seed: 随机种子
        
    Returns:
        OptimizationResult
    """
    from pdu.api import detonation_forward
    
    key = random.PRNGKey(seed)
    n_comp = len(available_components)
    
    # 生成密度网格
    densities = jnp.linspace(density_range[0], density_range[1], n_density_points)
    
    # 生成组成样本 (使用 Dirichlet 分布)
    key, subkey = random.split(key)
    alphas = jnp.ones(n_comp)
    compositions = random.dirichlet(subkey, alphas, shape=(n_composition_samples,))
    
    best_loss = float('inf')
    best_result = None
    
    for density in densities:
        for comp in compositions:
            # 计算性能
            result = detonation_forward(
                available_components,
                list(comp),
                float(density),
                verbose=False
            )
            
            perf = {'D': result.D, 'P_cj': result.P_cj, 'Q': result.Q}
            loss = multi_objective_loss(perf, targets)
            
            if loss < best_loss:
                best_loss = loss
                best_result = (comp, float(density), perf)
    
    if best_result is not None:
        comp, density, perf = best_result
        recipe = {
            name: float(frac)
            for name, frac in zip(available_components, comp)
            if frac > 0.01
        }
        total = sum(recipe.values())
        recipe = {k: v / total for k, v in recipe.items()}
        
        return OptimizationResult(
            recipe=recipe,
            density=density,
            performance=perf,
            loss=best_loss,
            converged=True,
            n_iters=n_density_points * n_composition_samples,
            loss_history=[]
        )
    
    return OptimizationResult(
        recipe={},
        density=0.0,
        performance={},
        loss=float('inf'),
        converged=False,
        n_iters=0,
        loss_history=[]
    )
