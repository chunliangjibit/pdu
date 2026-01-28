"""
感度估计与氧平衡模块

提供基于 Kamlet-Jacobs 公式的撞击感度估计和氧平衡计算。
"""

import jax
import jax.numpy as jnp
from typing import Dict


def compute_oxygen_balance(formula: Dict[str, int], molecular_weight: float = None) -> float:
    """计算氧平衡 OB%
    
    OB% = (1600/M) * (n_O - 2*n_C - 0.5*n_H - ...)
    
    假设碳完全氧化为 CO2，氢氧化为 H2O。
    
    Args:
        formula: 元素组成字典 {'C': x, 'H': y, 'N': z, 'O': w, ...}
        molecular_weight: 分子量 (g/mol)，如果为 None 则自动计算
        
    Returns:
        氧平衡 OB% (负值表示缺氧，正值表示富氧)
    """
    # 元素原子量
    atomic_weights = {
        'C': 12.011, 'H': 1.008, 'N': 14.007, 'O': 15.999,
        'Cl': 35.453, 'Al': 26.982, 'Mg': 24.305, 'B': 10.811,
        'K': 39.098, 'S': 32.065, 'F': 18.998
    }
    
    # 自动计算分子量
    if molecular_weight is None:
        molecular_weight = sum(
            formula.get(elem, 0) * atomic_weights.get(elem, 0) 
            for elem in formula
        )
    
    if molecular_weight <= 0:
        return 0.0
    
    # 获取各元素数目
    n_C = formula.get('C', 0)
    n_H = formula.get('H', 0)
    n_O = formula.get('O', 0)
    n_N = formula.get('N', 0)
    n_Al = formula.get('Al', 0)
    n_Cl = formula.get('Cl', 0)
    n_Mg = formula.get('Mg', 0)
    n_B = formula.get('B', 0)
    
    # 氧平衡计算
    # C -> CO2 (需要 2 个 O)
    # H -> H2O (需要 0.5 个 O)
    # Al -> Al2O3 (需要 1.5 个 O)
    # Mg -> MgO (需要 1 个 O)
    # B -> B2O3 (需要 1.5 个 O)
    # Cl -> HCl (释放 0.5 个 O 的等效)
    
    oxygen_needed = (2 * n_C + 0.5 * n_H + 1.5 * n_Al + 
                     1.0 * n_Mg + 1.5 * n_B - 0.5 * n_Cl)
    oxygen_balance = n_O - oxygen_needed
    
    OB_percent = (1600.0 / molecular_weight) * oxygen_balance
    
    return OB_percent


def compute_mixture_oxygen_balance(
    component_names: list,
    mass_fractions: list,
    component_data: dict
) -> float:
    """计算混合物的氧平衡
    
    Args:
        component_names: 组分名称列表
        mass_fractions: 质量分数列表
        component_data: 组分数据字典
        
    Returns:
        混合物氧平衡 OB%
    """
    total_ob = 0.0
    
    for name, frac in zip(component_names, mass_fractions):
        if name in component_data:
            comp = component_data[name]
            ob = compute_oxygen_balance(comp.formula, comp.molecular_weight)
        else:
            ob = 0.0
        total_ob += frac * ob
    
    return total_ob


@jax.jit
def estimate_impact_sensitivity(
    detonation_velocity: float,
    crystal_density: float
) -> float:
    """基于 Kamlet-Jacobs 公式估计撞击感度 h50
    
    经验关系式 (近似):
    log10(h50) ≈ a - b * (ρ * D²)
    
    其中 D 为爆速 (km/s)，ρ 为密度 (g/cm³)
    
    注意：这只是粗略估计，实际感度需要实验测定。
    
    Args:
        detonation_velocity: 爆速 (m/s)
        crystal_density: 晶体密度 (g/cm³)
        
    Returns:
        估计的 h50 (cm)，负值表示高感度，正值表示低感度
    """
    # 转换单位: m/s -> km/s
    D_km_s = detonation_velocity / 1000.0
    
    # 经验系数 (基于文献拟合)
    # 这些系数需要根据实际数据调整
    a = 3.5  # 截距
    b = 0.018  # 斜率
    
    # 计算 ρD²
    rho_D2 = crystal_density * D_km_s ** 2
    
    # log(h50) 估计
    log_h50 = a - b * rho_D2
    
    # h50 (cm)
    h50 = 10.0 ** log_h50
    
    # 限制在合理范围 [1, 300] cm
    h50 = jnp.clip(h50, 1.0, 300.0)
    
    return h50


def estimate_sensitivity_class(h50: float) -> str:
    """根据 h50 值评估感度等级
    
    Args:
        h50: 撞击感度 (cm)
        
    Returns:
        感度等级字符串
    """
    if h50 < 10:
        return "高感度 (敏感)"
    elif h50 < 30:
        return "中高感度"
    elif h50 < 80:
        return "中等感度"
    elif h50 < 150:
        return "低感度 (钝感)"
    else:
        return "极低感度 (极钝感)"


def kamlet_jacobs_detonation(
    formula: Dict[str, int],
    heat_of_formation: float,
    density: float
) -> Dict[str, float]:
    """Kamlet-Jacobs 经验公式估计爆轰性能
    
    标准 K-J 公式:
    D = A * φ^(1/2) * (1 + B*ρ₀)  [km/s]
    P_cj = K * ρ₀² * φ  [GPa]
    
    其中 φ = N * M^(1/2) * Q^(1/2)
    N: 每克炸药产生的气体摩尔数 (mol/g)
    M: 气态产物平均分子量 (g/mol)
    Q: 爆热 (cal/g)
    
    对于 CHNO 炸药:
    - 碳优先生成 CO2 (富氧) 或 CO + C (缺氧)
    - 氢生成 H2O
    - 氮生成 N2
    
    Args:
        formula: 元素组成字典
        heat_of_formation: 炸药生成热 (kJ/mol)
        density: 装药密度 (g/cm³)
        
    Returns:
        包含 D, P_cj, Q, OB 的字典
    """
    n_C = formula.get('C', 0)
    n_H = formula.get('H', 0)
    n_N = formula.get('N', 0)
    n_O = formula.get('O', 0)
    n_Al = formula.get('Al', 0)
    
    # 计算炸药分子量 (g/mol)
    M_exp = 12.011 * n_C + 1.008 * n_H + 14.007 * n_N + 15.999 * n_O + 26.982 * n_Al
    if M_exp <= 0:
        M_exp = 100.0  # 默认值
    
    # 氧平衡
    OB = compute_oxygen_balance(formula, M_exp)
    
    # ===== 简化产物组成计算 (含铝修正) =====
    # 规则: 
    # 1. Al 抢夺氧生成 Al2O3 (固体)，释放巨大热量
    # 2. H 优先与剩余 O 结合生成 H2O
    # 3. 剩余 O 与 C 生成 CO2/CO
    
    # 铝的氧化: 2Al + 1.5O2 -> Al2O3
    n_Al2O3 = n_Al / 2.0
    O_needed_Al = 1.5 * n_Al
    
    # 扣除被 Al 消耗的氧
    O_curr = n_O - O_needed_Al
    
    # 如果氧不够 Al 用? (极端情况)
    if O_curr < 0:
        # Al 未完全氧化? 
        # 简单处理: 假设 Al 即使缺氧也能通过其他途径(如还原H2O)反应，或者仅部分反应。
        # KJ 主要是估算，且大多数实用含铝炸药 (Tritonal, PBXN-109) 都是富氧或近平衡的基体加 Al。
        # 为保持物理一致性，如果 O < 0，则限制 Al2O3 生成量
        n_Al2O3 = n_O / 3.0
        O_curr = 0.0
        # 剩余 Al 作为惰性? 暂忽略
    
    # 水的生成: H2O
    n_H2O = n_H / 2.0
    O_remaining = O_curr - n_H2O
    
    if O_remaining >= 2 * n_C:
        # 富氧: 全部生成 CO2
        n_CO2 = n_C
        n_CO = 0.0
        n_C_solid = 0.0
    elif O_remaining >= n_C:
        # 中等: 部分 CO2，部分 CO
        # n_CO2 + n_CO = n_C
        # 2*n_CO2 + 1*n_CO = O_remaining
        # => n_CO2 = O_remaining - n_C
        n_CO2 = max(0.0, O_remaining - n_C)
        n_CO = max(0.0, 2 * n_C - O_remaining)
        n_C_solid = 0.0
    else:
        # 缺氧: 只有 CO 和固体碳 (对于含铝炸药，通常基体也因为 Al 抢氧而变得更缺氧)
        # n_CO + n_C_solid = n_C
        # 1*n_CO = O_remaining
        n_CO = max(0.0, O_remaining)
        n_CO2 = 0.0
        n_C_solid = max(0.0, n_C - O_remaining)
    
    # 氮气生成
    n_N2 = n_N / 2.0
    
    # ===== K-J 参数计算 =====
    # N: 每克炸药产生的气体产物摩尔数 (mol/g) (不含 Al2O3, C_solid)
    n_gas_total = n_H2O + n_CO2 + n_CO + n_N2
    N = n_gas_total / M_exp  # mol/g
    
    # M: 气态产物平均分子量 (g/mol)
    if n_gas_total > 0:
        M_gas = (18.015 * n_H2O + 44.009 * n_CO2 + 28.010 * n_CO + 28.014 * n_N2) / n_gas_total
    else:
        M_gas = 28.0
    
    # Q: 爆热 (cal/g)
    # Q ≈ -ΔHf(炸药)/M + Σ(n_i * ΔHf_prod_i) / M
    # ΔHf (kcal/mol): 
    # H2O(g)=-57.8, CO2=-94.1, CO=-26.4
    # Al2O3(s)=-400.0 (非常大!)
    Q_products = (n_H2O * 57.8 + n_CO2 * 94.1 + n_CO * 26.4 + n_Al2O3 * 400.0)  # kcal/mol
    Q_explosive = heat_of_formation / 4.184  # kJ/mol -> kcal/mol
    Q_total = (Q_products - Q_explosive) / M_exp * 1000  # cal/g
    Q_total = max(Q_total, 500)  # 最小爆热约束
    
    # ===== 标准 K-J 计算 =====
    # K-J 经验常数 (Kamlet & Jacobs, 1968)
    A = 1.01  # km/s
    B = 1.30  # cm³/g
    # K 值经过校准以匹配实验数据
    # 原始 Kamlet-Jacobs K = 15.58，但需要除以 phi 的量纲因子
    K = 1.558  # 校准值
    
    # φ 参数 (特征参数)
    # 单位: N(mol/g) * M^0.5(g/mol)^0.5 * Q^0.5(cal/g)^0.5
    phi = N * (M_gas ** 0.5) * (Q_total ** 0.5)
    
    # 爆速 D (km/s)
    D_kms = A * (phi ** 0.5) * (1 + B * density)
    D = D_kms * 1000  # m/s
    
    # 爆压 P_cj (GPa)
    # 标准公式: P_cj = K * ρ₀² * φ
    P_cj = K * (density ** 2) * phi
    
    # 放宽到合理范围
    D = max(min(D, 10000), 4000)  # 4-10 km/s
    P_cj = max(min(P_cj, 60), 5)  # 5-60 GPa
    
    return {
        'D': D,
        'P_cj': P_cj,
        'phi': phi,
        'Q_est': Q_total,
        'OB': OB,
        'N': N,
        'M_gas': M_gas
    }
