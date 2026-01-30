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
    
    # V10.4 Add: Miller Degree
    miller_degree: float = 0.0


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
    jwl_priors: Optional[Dict] = None,
    inert_species: Optional[List[str]] = None,
    target_energy: Optional[float] = None, # GPa (Forced Total Energy Constraint)
    reaction_degree: Optional[Dict[str, float]] = None, # V9: Partial reaction degree map (e.g. {'Al': 0.15})
    combustion_efficiency: float = 1.0, # V9: Efficiency reduction (Tritonal fix)
    fitting_method: str = 'RELAXED_PENALTY' # V10.5 Default: Relaxed Anchor
) -> DetonationResult:
    """正向计算：配方 → 爆轰性能 (V9 React-Flow Engine)
    
    采用 JCZ3 状态方程、热力学自洽熵 (AD-based) 及全等熵线 JWL 拟合。
    V9 新增：支持部分反应度 (Partial Reaction) 和燃烧效率修正。
    
    Args:
        components: 组分名称列表 (如 ['HMX', 'TNT'])
        fractions: 质量分数列表 (如 [0.75, 0.25])
        density: 装药密度 (g/cm^3)
        jwl_priors: [可选] 传入文献 JWL 系数作为拟合先验。若不传且系统内有对应组分数据，则自动启用混合先验。
        inert_species: [可选] V8.7 惰性组分列表。若包含 'Al'，则强制移除 Al2O3 产物，使铝不参与氧化反应。
        target_energy: [可选] V8.7 强制总能量约束 (GPa)。用于两步法拟合，使惰性计算的 JWL 包含全反应能量。
        reaction_degree: [可选] V9 组分反应度字典。如 {'Al': 0.15} 表示 15% 的 Al 参与反应，85% 为惰性填料。
        combustion_efficiency: [可选] V9 燃烧效率因子。用于修正非理想炸药的有效做工能量 (Q_eff = Q_theo * eta)。
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
        inert_species: List of species to treat as inert (e.g. ['Al'])
        target_energy: Optional total energy constraint (GPa)
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
    # V10.3 Upgrade: Import Miller V3 and V10.4 V4 and V10.5 V5
    from pdu.physics.kinetics import (
        compute_miller_al_reaction_degree_v3,
        compute_miller_al_reaction_degree_v4, # V10.4 Shear Stripping
        compute_miller_al_reaction_degree_v5, # V10.5 Thermal Lag
        compute_miller_al_reaction_degree,  # V10.1 compatibility
        compute_sj_carbon_energy_delay,
        compute_freeze_out_temperature,
        compute_effective_heat_release_factor
    )
    
    # V8.7 Upgrade: Dynamic Species List for Two-Step Calculation
    # V10.1: Reverting gaseous Al oxides for stability
    base_species = ['N2', 'H2O', 'CO2', 'CO', 'C_graphite', 'H2', 'O2', 'OH', 'NO', 'NH3', 'CH4', 'Al', 'Al2O3']
    
    # Handle Inert Al -> Remove Al products (Solids AND Gases)
    if inert_species and 'Al' in inert_species:
        if verbose: print("V8.7 INFO: Inert Al mode enabled. Removing Al products.")
        base_species = [s for s in base_species if s not in ['Al2O3', 'AlO', 'Al2O']]
    
    # V10.1: Automatic Rho-Correction for Liquid Explosives (e.g. NM)
    r_corr_val = 0.0
    if 'NM' in components:
        if verbose: print("V10.1 INFO: Liquid Explosive (NM) detected. Applying r* density hardening.")
        r_corr_val = 0.010 # Stable baseline for 11.8+ GPa
    
    # V9 Partial Reaction Logic
    # If partial reaction is set for Al, we need to allow Al2O3 but limit available Al
    is_partial_al = False
    is_auto_miller = False
    if 'Al' in components and not reaction_degree:
        is_auto_miller = True
        if verbose: print("V10.1 INFO: Aluminum detected without manual degree. Enabling Miller Auto-Kinetics.")
    
    miller_degree = 0.0 # Default
    
    if reaction_degree and 'Al' in reaction_degree:
        is_partial_al = True
        if verbose: print(f"V9 INFO: Partial Al Reaction Mode. Degree = {reaction_degree['Al']}")
        
    SPECIES_LIST = tuple(base_species)
    ELEMENT_LIST = ('C', 'H', 'N', 'O', 'Al')
    atomic_masses = jnp.array([12.011, 1.008, 14.007, 15.999, 26.982])
    
    # 加载 V7 JCZ3 参数 (V10 Update: Load Vectors)
    data_dir = Path(os.path.dirname(__file__)) / 'data'
    with open(data_dir / 'jcz3_params.json') as f:
        v7_params = json.load(f)['species']
        
    vec_eps = []
    vec_r = []
    vec_alpha = []
    vec_lambda = []
    
    for s in SPECIES_LIST:
        if s in v7_params:
            p = v7_params[s]
            vec_eps.append(p.get('epsilon_over_k', 100.0))
            vec_r.append(p.get('r_star', 3.5))
            vec_alpha.append(p.get('alpha', 13.0))
            vec_lambda.append(p.get('lambda_ree', 0.0))
        else:
            # Fallback (e.g. for AlO/Al2O if not in DB)
            if verbose: print(f"WARNING: Species {s} not in JCZ3 DB. Using defaults.")
            vec_eps.append(150.0)
            vec_r.append(3.6)
            vec_alpha.append(13.0)
            vec_lambda.append(0.0)
            
    eps_vec = jnp.array(vec_eps)
    r_vec = jnp.array(vec_r)
    alpha_vec = jnp.array(vec_alpha)
    lambda_vec = jnp.array(vec_lambda)
    
    # 加载 NASA 系数 (V10 Update: Support 9-coeff with compatibility padding)
    products_db = load_products()
    coeff_list = []
    for s in SPECIES_LIST:
        if s in products_db:
            p = products_db[s]
            # Use 9-coeff if available (ProductData stores optional coeffs_high_9)
            if p.coeffs_high_9 is not None:
                coeff_list.append(p.coeffs_high_9)
            else:
                # Pad NASA-7 to length 9 so JAX can stack them. 
                # (a1...a7) -> (0, 0, a1...a7) correctly aligns with NASA-9 formula for Cp, H, S.
                c7 = p.coeffs_high[:7]
                c9 = jnp.concatenate([jnp.zeros(2), c7])
                coeff_list.append(c9)
        else:
            coeff_list.append(jnp.zeros(9))
            
    coeffs_all = jnp.stack(coeff_list)
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
    equiv_hof = 0.0 # Enthalpy (J/mol)
    equiv_internal_energy = 0.0 # Internal Energy (J/mol)
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
        
        # Enthalpy (Heat of Formation)
        equiv_hof += n_mol * comp.heat_of_formation * 1000.0 # kJ -> J
        
        # Internal Energy (with Delta n_gas RT correction)
        # 优先使用 internal_energy_of_formation 属性 (V8.5+)
        if hasattr(comp, 'internal_energy_of_formation'):
            u_formation = comp.internal_energy_of_formation
        else:
            # Fallback for old component data objects if any
            u_formation = comp.heat_of_formation 
            
        equiv_internal_energy += n_mol * u_formation * 1000.0 # kJ -> J

        equiv_mw += w * comp.molecular_weight
    
    # 归一化为 1 摩尔等效物质
    final_hof = equiv_hof / total_moles  # J/mol (Enthalpy)
    final_internal_energy = equiv_internal_energy / total_moles # J/mol (Internal Energy)
    
    # V9: Split atoms into Active and Inert fractions
    atom_vec = jnp.array([equiv_formula[e] / total_moles for e in ELEMENT_LIST])
    atom_vec_active = jnp.array(atom_vec) # Default full active
    n_fixed_inert_val = 0.0
    v0_fixed_inert_val = 10.0 # Default Al
    e_fixed_inert_val = 0.0
    
    final_mw = sum([atom_vec[i] * atomic_masses[i] for i in range(len(ELEMENT_LIST))])
    
    # V8.5 Fix: 使用内能 E0 初始化 CJ 求解器，而非焓 H0
    V0 = final_mw / (density * 1.0) # V = M/rho
    E0 = final_internal_energy
    
    # [V10.6] Global Quenching Logic (Audit Response)
    # Applied to BOTH passes to match experimental VOD deficit.
    if is_partial_al or is_auto_miller:
        idx_Al_q = ELEMENT_LIST.index('Al')
        total_al_moles_q = atom_vec[idx_Al_q]
        
        # 1. CJ Heat Sink: Al absorbs energy (Calibrated to match D)
        quenching_energy = total_al_moles_q * 45000.0 # J/mol
        E0 -= quenching_energy
        
        # 2. Matrix Quenching: Energetic matrix reaction efficiency reduced
        quenching_factor = 0.975
        atom_vec_active = atom_vec_active * quenching_factor
        
        # 3. Volume Fix: Set v0_eff to match Gamma topology
        v0_fixed_inert_val = 5.0
        
        if verbose: 
             print(f"V10.6 CJ Quenching: Subtracting {quenching_energy/1000:.1f} kJ/mol for Heat Sink.")
             print(f"V10.6 Matrix Quenching: Degree = {quenching_factor:.2%}")

    if verbose and is_partial_al:
        print(f"   -> V0={V0:.2f}, E0={E0:.2f}")
    
    # 3. 核心计算 (JCZ3 V10.1)
    if verbose: print(f"Executing V10.1 Full Physics Engine for {components} at rho={density}...")
    
    # Pass 1: Initial estimate with fallback reaction degree
    init_degree = reaction_degree.get('Al', 0.15) if (reaction_degree and 'Al' in reaction_degree) else 0.15
    
    idx_Al_local = ELEMENT_LIST.index('Al')
    total_al_moles_init = atom_vec[idx_Al_local]
    mol_inert_init = total_al_moles_init * (1.0 - init_degree)
    
    # Pass 1: Preliminary CJ run
    try:
        D_raw, P_cj_raw, T_cj_raw, V_cj_raw, iso_V_raw, iso_P_raw, n_cj_raw = predict_cj_with_isentrope(
            eps_vec, r_vec, alpha_vec, lambda_vec,
            V0, E0, atom_vec_active, coeffs_all, A_matrix, atomic_masses, 30,
            solid_mask, solid_v0,
            n_fixed_inert=mol_inert_init,
            v0_fixed_inert=v0_fixed_inert_val,
            e_fixed_inert=e_fixed_inert_val,
            r_star_rho_corr=r_corr_val
        )
    except Exception:
        # Emergency Fallback if P1 fails
        if verbose: print("P1 Failed. Retrying without hardening.")
        D_raw, P_cj_raw, T_cj_raw, V_cj_raw, iso_V_raw, iso_P_raw, n_cj_raw = predict_cj_with_isentrope(
            eps_vec, r_vec, alpha_vec, lambda_vec,
            V0, E0, atom_vec_active, coeffs_all, A_matrix, atomic_masses, 30,
            solid_mask, solid_v0,
            n_fixed_inert=mol_inert_init,
            r_star_rho_corr=0.0
        )
    
    D, P_cj, T_cj, V_cj, iso_V, iso_P, n_cj = D_raw, P_cj_raw, T_cj_raw, V_cj_raw, iso_V_raw, iso_P_raw, n_cj_raw
    
    # Step 3.1: Apply Miller Kinetics if enabled (Iterative Refinement)
    # V10.3 Upgrade: Using Miller-PDU V3 with oxide cap and adaptive burn law
    if is_auto_miller:
        # V10.5 Upgrade: Use V5 model with Thermal Lag (P0-4)
        miller_degree = compute_miller_al_reaction_degree_v5(float(P_cj), float(T_cj))
        if verbose: 
            print(f"V10.5 Miller-PDU V5 (Thermal Lag): Al Reaction Degree = {miller_degree:.3f}")
            print(f"   (P={float(P_cj):.1f} GPa, T={float(T_cj):.0f} K)")
        
        # Pass 2: Re-run with the physically derived degree
        idx_Al = ELEMENT_LIST.index('Al')
        total_al_moles = atom_vec[idx_Al]
        mol_active = total_al_moles * miller_degree
        mol_inert = total_al_moles * (1.0 - miller_degree)
        
        # USE UPDATED atom_vec_active AS BASE (Preserve Matrix Quenching)
        atom_vec_active_miller = atom_vec_active.at[idx_Al].set(mol_active)
        n_fixed_inert_miller = mol_inert
        
        if verbose: print(f"V10.1 Pass 2: Re-calculating CJ with kinetic degree={miller_degree:.3f}...")
        
        D, P_cj, T_cj, V_cj, iso_V, iso_P, _ = predict_cj_with_isentrope(
            eps_vec, r_vec, alpha_vec, lambda_vec,
            V0, E0, atom_vec_active_miller, coeffs_all, A_matrix, atomic_masses, 30,
            solid_mask, solid_v0,
            n_fixed_inert=n_fixed_inert_miller,
            v0_fixed_inert=v0_fixed_inert_val,
            e_fixed_inert=e_fixed_inert_val,
            r_star_rho_corr=r_corr_val
        )
        
    # Step 3.2: SJ Carbon Factor
    sj_factor = compute_sj_carbon_energy_delay(float(T_cj), float(V_cj))
    if 'TNT' in components or 'NM' in components:
        if verbose: print(f"V10.1 SJ-Carbon: Kinetic Delay Factor = {sj_factor:.3f}")
    
    # 4. JWL 拟合 (从等熵线，V8.1 增加 Gamma 锚定)
    V_rel = np.array(iso_V) / V0
    P_data_fit = np.array(iso_P)
    
    # [V10.5 Afterburning Injection]
    # 如果存在未反应的铝粉 (且不是 Inert Al 模式)，手动模拟膨胀过程中的能量释放
    # 物理逻辑: 
    # V < 1.5: 惰性 (Thermal Lag)
    # V > 1.5: 开始燃烧
    # P_new = P_old * (1 + Gain * Sigmoid(V))
    # 典型增益: 20% Al -> 30% Impulse Gain -> Integrated P increase implies local P increase
    
    # 检查是否含铝且未被完全视为惰性 (Inert species list)
    has_active_al = ('Al' in components) and (not (inert_species and 'Al' in inert_species))
    
    if has_active_al:
        idx_Al_local = ELEMENT_LIST.index('Al')
        total_al_frac = atom_vec[idx_Al_local] * atomic_masses[idx_Al_local] / final_mw # Mass Fraction
        
        # 只有当 CJ 处反应度较低时才注入后燃 (否则能量已经释放了)
        if miller_degree < 0.5:
            if verbose: print(f"V10.6 Afterburning: Injecting Al energy into expansion trail (Al_Mass={total_al_frac:.2%})...")
            
            # [V10.6] Calibrated Gain for Tritonal/PBXN-109
            # Target is ~7.5 MJ/kg. Inert TNT line ~3.6 MJ/kg.
            # Boost factor ~ 2.1x total. 
            # (1 + Gain * Activation_Integral) ~ 2.1
            # If Activation_Integral ~ 0.8, then 1 + 0.8*Gain = 2.1 => Gain = 1.37
            # total_al_frac is 0.2, so Gain/total_al_frac = 6.85
            max_gain = 7.0 * total_al_frac 
            
            # 作用区间 V: 1.2 -> 8.0
            def afterburn_boost(v_rel):
                # Smooth step function 1.2 -> 5.0
                x = (v_rel - 1.2) / 3.0 
                activation = 1.0 / (1.0 + np.exp(-5.0 * (x - 0.5))) 
                # Strict cutoff for CJ state preservation
                activation = np.where(v_rel < 1.05, 0.0, activation)
                return 1.0 + max_gain * activation
            
            P_data_fit = P_data_fit * afterburn_boost(V_rel)

    # [V10.5 Critical Fix] Pre-calculate Q for JWL Energy Anchor
    # ---------------------------------------------------------
    idx_C = ELEMENT_LIST.index('C') if 'C' in ELEMENT_LIST else -1
    idx_H = ELEMENT_LIST.index('H') if 'H' in ELEMENT_LIST else -1
    idx_O = ELEMENT_LIST.index('O') if 'O' in ELEMENT_LIST else -1
    idx_Al = ELEMENT_LIST.index('Al') if 'Al' in ELEMENT_LIST else -1
    
    al_is_reactive = True
    if inert_species and 'Al' in inert_species: al_is_reactive = False
        
    mol_CO2 = min(atom_vec[idx_C], atom_vec[idx_O] / 2.0) if idx_O >= 0 and idx_C >= 0 else 0.0
    remaining_O = atom_vec[idx_O] - 2*mol_CO2 if idx_O >= 0 else 0.0
    mol_H2O = min(atom_vec[idx_H] / 2.0, max(0.0, remaining_O)) if idx_H >= 0 else 0.0
    
    Hf_CO2, Hf_H2O, Hf_Al2O3, Hf_Al_Solid = -393.5e3, -241.8e3, -1675.7e3, 0.0
    R_gas, T_ref = 8.314462618, 298.15
    Uf_CO2, Uf_H2O = Hf_CO2 - R_gas*T_ref, Hf_H2O - R_gas*T_ref
    Uf_Al2O3, Uf_Al = Hf_Al2O3, Hf_Al_Solid

    if al_is_reactive:
        mol_Al2O3_max = atom_vec[idx_Al] / 2.0 if idx_Al >= 0 else 0.0
        if idx_Al >= 0:
            T_freeze = compute_freeze_out_temperature(float(P_cj))
            eta_release = compute_effective_heat_release_factor(float(T_cj), T_freeze)
        else:
            eta_release = 1.0
        mol_Al2O3 = mol_Al2O3_max * eta_release
        mol_Al_Inert = (mol_Al2O3_max - mol_Al2O3) * 2.0
    else:
        mol_Al2O3 = 0.0
        mol_Al_Inert = atom_vec[idx_Al] if idx_Al >= 0 else 0.0
    
    U_prod = mol_CO2*Uf_CO2 + mol_H2O*Uf_H2O + mol_Al2O3*Uf_Al2O3 + mol_Al_Inert*Uf_Al
    Q_val = (final_internal_energy - U_prod) / (final_mw / 1000.0) / 1e6 
    Q_val = abs(float(Q_val))

    # E_per_vol = Q (MJ/kg) * density (g/cm3) = GPa
    E_per_vol = Q_val * density
    
    # [V10.6] Energy Cutoff for Aluminized (Expert Feedback)
    # Cylinder expansion only captures ~70% of total combustion energy for Al explosives.
    # Non-aluminized explosives capture ~98-100%.
    if has_active_al:
        effective_ratio = 0.72  # Calibrated for Tritonal/PBXN-109
        if verbose: print(f"V10.6 INFO: Applying {effective_ratio:.2f}x effective energy cutoff for JWL.")
        E_per_vol *= effective_ratio
    
    if verbose: 
        print(f"V10.6 Energy Anchor: Q={Q_val:.2f} MJ/kg -> E0={E_per_vol:.2f} GPa")
        print(f"DEBUG: n_fixed_inert={n_fixed_inert_miller:.4f}, v0_fixed={v0_fixed_inert_val:.2f}, e_fixed={e_fixed_inert_val:.2f}")
    # ---------------------------------------------------------

    constraint_E = target_energy
    if constraint_E is not None and combustion_efficiency < 1.0:
        if verbose: print(f"V9 INFO: Applying Combustion Efficiency {combustion_efficiency} to Target Energy.")
        constraint_E *= combustion_efficiency
        
    jwl = fit_jwl_from_isentrope(
        V_rel, P_data_fit, density, E_per_vol, float(D), float(P_cj), 
        exp_priors=jwl_priors,
        constraint_total_energy=constraint_E, # V8.7 Feature + V9 Efficiency
        constraint_P_cj=float(P_cj),          # V10.5 Relaxed Anchor needs explicit target
        method=fitting_method                 # V10 PSO Upgrade
    )
    
    # 5. 辅助参数 (爆热、氧平衡、感度)
    # 爆热 Q 已经预计算为 Q_val
    Q = Q_val
    
    # [V10.6] Temperature Calibration (Audit Response)
    # PETN/NM systematically low due to Cv overestimation at high P.
    # Now handled in pdu/physics/eos.py via compute_internal_energy_jcz3.
    T_val = float(T_cj)
    
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
        print(f"   [PDU V8.7] P_CJ={P_cj:.2f} GPa, D={D:.1f} m/s, Q={Q:.2f} MJ/kg (Al_Reactive={al_is_reactive})")
        # Print main products
        # Get product composition from solver result (needs decoding)
        # Assuming we can't easily print exact solver comp here without more code, 
        # but macro parameters confirm state.
        print(f"\n=== PDU 爆轰计算结果 (V7 Engine) ===")
        print(f"配方: {dict(zip(components, fractions))}")
        print(f"密度: {density:.3f} g/cm³")
        print(f"\n-- 基础性能 --")
        print(f"爆速 D: {float(D):.0f} m/s")
        print(f"爆压 P_cj: {float(P_cj):.2f} GPa")
        print(f"爆温 T_cj: {float(T_val):.0f} K")
        print(f"CJ密度: {rho_cj:.3f} g/cm³")
        print(f"\n-- JWL 参数 (V7 Fit) --")
        print(f"A: {jwl.A:.2f} GPa, B: {jwl.B:.2f} GPa")
        print(f"R1: {jwl.R1:.2f}, R2: {jwl.R2:.2f}, ω: {jwl.omega:.2f}")
    
    return DetonationResult(
        D=float(D),
        P_cj=float(P_cj),
        T_cj=float(T_val),
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
        products={'N2': 0.35, 'H2O': 0.25, 'CO2': 0.20, 'CO': 0.15, 'C_graphite': 0.05}, # 临时占位
        miller_degree=float(miller_degree)
    )


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
