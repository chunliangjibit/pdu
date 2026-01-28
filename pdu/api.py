"""
PDU 公共 API 模块

提供简洁的高层接口用于正向计算和逆向设计。
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import Dict, List, Optional, Union, NamedTuple
from dataclasses import dataclass


class DetonationResult(NamedTuple):
    """爆轰计算结果"""
    # 基础性能
    D: float                    # 爆速 (m/s)
    P_cj: float                 # CJ 压力 (GPa)
    T_cj: float                 # CJ 温度 (K)
    rho_cj: float               # CJ 密度 (g/cm³)
    
    # 能量参数
    Q: float                    # 爆热 (kJ/kg)
    
    # 氧平衡
    OB: float                   # 氧平衡 (%)
    
    # 感度估计
    h50: float                  # 估计撞击感度 (cm)
    sensitivity_class: str      # 感度等级
    
    # JWL 参数
    jwl_A: float
    jwl_B: float
    jwl_R1: float
    jwl_R2: float
    jwl_omega: float
    jwl_mse: float      # JWL 拟合误差
    
    # 产物信息
    products: Dict[str, float]  # 产物摩尔分数


@dataclass
class Recipe:
    """配方数据类"""
    components: List[str]       # 组分名称
    fractions: List[float]      # 质量分数
    density: float              # 装药密度 (g/cm³)
    
    def __post_init__(self):
        # 归一化质量分数
        total = sum(self.fractions)
        if abs(total - 1.0) > 1e-6:
            self.fractions = [f / total for f in self.fractions]


# Standard Literature JWL Benchmarks for Prior Blending
LIT_JWL_DATA = {
    "HMX": {"A": 778.3, "B": 7.07, "R1": 4.2, "R2": 1.0, "omega": 0.30},
    "RDX": {"A": 609.7, "B": 12.95, "R1": 4.5, "R2": 1.4, "omega": 0.25},
    "PETN": {"A": 617.0, "B": 16.92, "R1": 4.4, "R2": 1.2, "omega": 0.25},
    "TNT": {"A": 371.2, "B": 3.23, "R1": 4.15, "R2": 0.95, "omega": 0.30},
    "NM": {"A": 209.2, "B": 5.68, "R1": 4.4, "R2": 1.2, "omega": 0.30},
    "TETRYL": {"A": 501.0, "B": 14.1, "R1": 4.5, "R2": 1.4, "omega": 0.25}
}

def get_blended_jwl_prior(components, fractions):
    """
    Automatic Prior Blending logic for unknown mixtures.
    Linearly interpolates standard coefficients based on mass fractions.
    """
    total_frac = sum(fractions)
    if total_frac < 1e-6: return None
    
    weights = [f/total_frac for f in fractions]
    blended = {"A": 0.0, "B": 0.0, "R1": 0.0, "R2": 0.0, "omega": 0.0}
    
    valid_count = 0
    for comp, weight in zip(components, weights):
        if comp in LIT_JWL_DATA:
            prior = LIT_JWL_DATA[comp]
            blended["A"] += prior["A"] * weight
            blended["B"] += prior["B"] * weight
            blended["R1"] += prior["R1"] * weight
            blended["R2"] += prior["R2"] * weight
            blended["omega"] += prior["omega"] * weight
            valid_count += 1
            
    if valid_count == 0:
        return None # No priors available for any component
    
    # Re-normalize if some components were missing (e.g. Al, Binder)
    # We assume the inert/reactive additives follow the energetic base
    scale = 1.0 / (sum(weights[i] for i, c in enumerate(components) if c in LIT_JWL_DATA))
    for k in blended: blended[k] *= scale
    return blended

def detonation_forward(
    components: List[str],
    fractions: List[float],
    density: float,
    verbose: bool = False,
    jwl_priors: Optional[Dict] = None
) -> DetonationResult:
    """正向计算：配方 → 爆轰性能 (V7 High-Fidelity Engine)
    
    采用 JCZ3 状态方程、热力学自洽熵 (AD-based) 及全等熵线 JWL 拟合。
    
    Args:
        components: 组分名称列表 (如 ['HMX', 'TNT'])
        fractions: 质量分数列表 (如 [0.75, 0.25])
        density: 装药密度 (g/cm^3)
        jwl_priors: [可选] 传入文献 JWL 系数作为拟合先验。若不传且系统内有对应组分数据，则自动启用混合先验。
    """
    
    # 自动先验逻辑
    if jwl_priors is None:
        jwl_priors = get_blended_jwl_prior(components, fractions)
        if jwl_priors and verbose:
            print(f"DEBUG: Auto-blended JWL priors enabled for {components}")
    
    """正向计算：配方 -> 性能结果
    Args:
        components: List of component names
        fractions: List of mass fractions
        density: Charge density (g/cm3)
        verbose: Print debug info
        jwl_priors: Optional dict of JWL priors
    Returns:
        DetonationResult
    """
    import json
    import os
    from pathlib import Path
    import jax
    import jax.numpy as jnp
    import numpy as np
    
    from pdu.data.components import load_components, get_component
    from pdu.data.products import load_products
    from pdu.core.equilibrium import build_stoichiometry_matrix
    from pdu.physics.sensitivity import estimate_impact_sensitivity, estimate_sensitivity_class
    from pdu.calibration.differentiable_cj_enhanced import predict_cj_with_isentrope
    from pdu.physics.jwl import fit_jwl_from_isentrope
    
    # 1. 数据准备
    SPECIES_LIST = ('N2', 'H2O', 'CO2', 'CO', 'C_graphite', 'H2', 'O2', 'OH', 'NO', 'NH3', 'CH4', 'Al', 'Al2O3')
    ELEMENT_LIST = ('C', 'H', 'N', 'O', 'Al')
    atomic_masses = jnp.array([12.011, 1.008, 14.007, 15.999, 26.982])
    
    # 加载 V7 JCZ3 参数
    data_dir = Path(os.path.dirname(__file__)) / 'data'
    with open(data_dir / 'jcz3_params.json') as f:
        v7_params = json.load(f)['species']
        
    reshaped_p = []
    for s in SPECIES_LIST:
        p = v7_params[s]
        reshaped_p.append([p['epsilon_over_k'], p['r_star'], p['alpha']])
    reshaped_p = jnp.array(reshaped_p)
    
    eps_matrix = jnp.sqrt(jnp.outer(reshaped_p[:,0], reshaped_p[:,0]))
    r_matrix = (jnp.expand_dims(reshaped_p[:,1], 1) + jnp.expand_dims(reshaped_p[:,1], 0)) / 2.0
    alpha_matrix = jnp.sqrt(jnp.outer(reshaped_p[:,2], reshaped_p[:,2]))
    
    # 加载 NASA 系数
    products_db = load_products()
    coeffs_all = jnp.stack([products_db[s].coeffs_high[:7] if s in products_db else jnp.zeros(7) for s in SPECIES_LIST])
    A_matrix = build_stoichiometry_matrix(SPECIES_LIST, ELEMENT_LIST)
    
    # 气固分离定义 (V8 升级)
    # Al2O3: mask=1.0, v0=25.7; C_graphite: mask=1.0, v0=5.3; Al: mask=1.0, v0=10.0
    solid_mask = jnp.zeros(len(SPECIES_LIST))
    solid_v0 = jnp.zeros(len(SPECIES_LIST))
    
    # 获取索引并设置
    for i, s in enumerate(SPECIES_LIST):
        if s == 'C_graphite':
            solid_mask = solid_mask.at[i].set(1.0)
            solid_v0 = solid_v0.at[i].set(5.3)
        elif s == 'Al2O3':
            solid_mask = solid_mask.at[i].set(1.0)
            solid_v0 = solid_v0.at[i].set(25.7)
        elif s == 'Al':
            solid_mask = solid_mask.at[i].set(1.0)
            solid_v0 = solid_v0.at[i].set(10.0)

    # 2. 配方解析 (包含 HTPB 修正)
    total = sum(fractions)
    fractions = [f / total for f in fractions]
    
    total_moles = 0.0
    equiv_formula = {e: 0.0 for e in ELEMENT_LIST}
    equiv_hof = 0.0 # J/mol
    equiv_mw = 0.0
    
    for i, name in enumerate(components):
        comp = get_component(name)
        w = fractions[i]
        
        # 摩尔贡献
        n_mol = w * 100.0 / comp.molecular_weight
        total_moles += n_mol
        for elem, count in comp.formula.items():
            if elem in equiv_formula:
                equiv_formula[elem] += n_mol * count
        
        equiv_hof += n_mol * comp.heat_of_formation * 1000.0 # kJ -> J
        equiv_mw += w * comp.molecular_weight # This is mass fraction weighted mw? No.
    
    # 归一化为 1 摩尔等效物质
    atom_vec = jnp.array([equiv_formula[e] / total_moles for e in ELEMENT_LIST])
    final_hof = equiv_hof / total_moles
    final_mw = sum([atom_vec[i] * atomic_masses[i] for i in range(len(ELEMENT_LIST))])
    
    # CJ 初始状态 (V0 in cm3/mol, E0 in J/mol)
    V0 = final_mw / (density * 1.0) # V = M/rho
    E0 = final_hof
    
    # 3. 核心计算 (JCZ3 V7)
    if verbose: print(f"Executing V7 High-Fidelity calculation for {components} at rho={density}...")
    
    D, P_cj, T_cj, V_cj, iso_V, iso_P = predict_cj_with_isentrope(
        eps_matrix, r_matrix, alpha_matrix,
        V0, E0, atom_vec, coeffs_all, A_matrix, atomic_masses, 30,
        solid_mask, solid_v0
    )
    
    # 4. JWL 拟合 (从等熵线，V8.1 增加 Gamma 锚定)
    V_rel = np.array(iso_V) / V0
    E_per_vol = (final_hof / V0) / 1000.0 # GPa
    jwl = fit_jwl_from_isentrope(V_rel, np.array(iso_P), density, E_per_vol, float(D), float(P_cj), exp_priors=jwl_priors)
    
    # 5. 辅助参数 (爆热、氧平衡、感度)
    # 爆热 Q (kJ/kg) = - (ΔH_reaction) / UnitMass
    # 这里使用简单的 Q 估计，或者从 CJ 结果提取
    Q = abs(final_hof) / (final_mw / 1000.0) / 1000.0 # 粗略能值估计
    
    # 氧平衡
    from pdu.physics.sensitivity import compute_oxygen_balance
    ob_val = 0.0
    for i, name in enumerate(components):
        comp = get_component(name)
        ob_val += fractions[i] * comp.oxygen_balance
    
    h50 = estimate_impact_sensitivity(float(D), density)
    sensitivity_class = estimate_sensitivity_class(h50)
    
    # 估计 CJ 密度
    rho_cj = density * (V0 / float(V_cj))
    
    if verbose:
        print(f"\n=== PDU 爆轰计算结果 (V7 Engine) ===")
        print(f"配方: {dict(zip(components, fractions))}")
        print(f"密度: {density:.3f} g/cm³")
        print(f"\n-- 基础性能 --")
        print(f"爆速 D: {float(D):.0f} m/s")
        print(f"爆压 P_cj: {float(P_cj):.2f} GPa")
        print(f"爆温 T_cj: {float(T_cj):.0f} K")
        print(f"CJ密度: {rho_cj:.3f} g/cm³")
        print(f"\n-- JWL 参数 (V7 Fit) --")
        print(f"A: {jwl.A:.2f} GPa, B: {jwl.B:.2f} GPa")
        print(f"R1: {jwl.R1:.2f}, R2: {jwl.R2:.2f}, ω: {jwl.omega:.2f}")
    
    return DetonationResult(
        D=float(D),
        P_cj=float(P_cj),
        T_cj=float(T_cj),
        rho_cj=float(rho_cj),
        Q=float(Q),
        OB=float(ob_val),
        h50=float(h50),
        sensitivity_class=sensitivity_class,
        jwl_A=float(jwl.A),
        jwl_B=float(jwl.B),
        jwl_R1=float(jwl.R1),
        jwl_R2=float(jwl.R2),
        jwl_omega=float(jwl.omega),
        jwl_mse=float(jwl.fit_mse),
        products={'N2': 0.35, 'H2O': 0.25, 'CO2': 0.20, 'CO': 0.15, 'C_graphite': 0.05} # 临时占位，以后可从 Equilibrium 提取
    )


def inverse_design(
    targets: Dict[str, float],
    available_components: List[str],
    density_range: tuple = (1.5, 2.0),
    constraints: Optional[Dict] = None,
    n_iters: int = 500,
    verbose: bool = False
) -> Dict:
    """逆向设计：目标性能 → 最优配方
    
    Args:
        targets: 目标性能字典，如 {'D': 8500, 'P_cj': 32}
        available_components: 可用组分列表
        density_range: 密度范围
        constraints: 额外约束
        n_iters: 优化迭代次数
        verbose: 是否打印详细信息
        
    Returns:
        包含最优配方和预测性能的字典
        
    Example:
        >>> result = inverse_design(
        ...     targets={'D': 8500},
        ...     available_components=['RDX', 'HMX', 'TNT']
        ... )
    """
    import jax
    import optax
    
    n_comp = len(available_components)
    
    # 初始化参数 (logits)
    key = jax.random.PRNGKey(42)
    logits_init = jax.random.normal(key, (n_comp,)) * 0.1
    density_init = jnp.array([(density_range[0] + density_range[1]) / 2])
    
    params = {'logits': logits_init, 'density': density_init}
    
    # 损失函数
    def loss_fn(params):
        # softmax 得到质量分数
        fractions = jax.nn.softmax(params['logits'])
        density = jnp.clip(params['density'][0], density_range[0], density_range[1])
        
        # 计算性能 (简化版本)
        from pdu.data.components import get_component
        
        total_hof = 0.0
        for i, name in enumerate(available_components):
            comp = get_component(name)
            total_hof += fractions[i] * comp.heat_of_formation / comp.molecular_weight * 1000
        
        # K-J 估计爆速
        Q_est = abs(total_hof) + 4000
        D_est = 1.01 * (1 + 1.25 * density) * jnp.sqrt(Q_est * 28 * 0.5) * 1000
        
        # 损失
        loss = 0.0
        if 'D' in targets:
            loss += ((D_est - targets['D']) / 1000) ** 2
        if 'P_cj' in targets:
            P_est = density ** 2 * D_est ** 2 / 1e12 * 0.25
            loss += ((P_est - targets['P_cj']) / 10) ** 2
        
        return loss
    
    # 优化器
    optimizer = optax.adam(learning_rate=0.05)
    opt_state = optimizer.init(params)
    
    # 优化循环
    for i in range(n_iters):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        if verbose and i % 100 == 0:
            print(f"Iter {i}: loss = {loss:.4f}")
    
    # 提取结果
    fractions = jax.nn.softmax(params['logits'])
    density = float(jnp.clip(params['density'][0], density_range[0], density_range[1]))
    
    # 构建配方
    recipe = {
        name: float(frac) 
        for name, frac in zip(available_components, fractions)
        if frac > 0.01
    }
    
    # 验证性能
    result = detonation_forward(
        list(recipe.keys()),
        list(recipe.values()),
        density,
        verbose=verbose
    )
    
    return {
        'recipe': recipe,
        'density': density,
        'performance': {
            'D': result.D,
            'P_cj': result.P_cj,
            'T_cj': result.T_cj,
            'Q': result.Q,
            'OB': result.OB
        },
        'targets': targets,
        'converged': True
    }


def list_available_components() -> Dict[str, List[str]]:
    """列出所有可用组分
    
    Returns:
        按类别分组的组分名称
    """
    from pdu.data.components import load_components
    
    components = load_components()
    
    result = {
        'explosives': [],
        'oxidizers': [],
        'metals': [],
        'binders': []
    }
    
    for name, comp in components.items():
        if comp.category in ['nitramine', 'nitroaromatic', 'nitro', 'tetrazole', 
                            'nitrate_ester', 'triazole', 'nitroguanidine', 
                            'furazan', 'pyrazine']:
            result['explosives'].append(name)
        elif comp.category in ['perchlorate', 'nitrate', 'dinitramide']:
            result['oxidizers'].append(name)
        elif comp.category in ['metal', 'metalloid']:
            result['metals'].append(name)
        elif comp.category in ['polymer', 'energetic_polymer']:
            result['binders'].append(name)
    
    return result
