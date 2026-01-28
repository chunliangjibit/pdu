"""
动态掩码工具模块

生成和管理活性物种掩码，用于化学平衡计算中排除不可能的产物。
"""

import jax.numpy as jnp
from typing import List, Dict, Set


def generate_active_mask(
    species_list: List[str],
    available_elements: Set[str]
) -> jnp.ndarray:
    """生成活性物种掩码
    
    根据可用元素生成掩码，排除不可能生成的产物。
    
    Args:
        species_list: 候选物种列表
        available_elements: 可用元素集合
        
    Returns:
        掩码数组 (1.0 表示活性, 0.0 表示非活性)
    """
    # 物种元素组成
    species_elements = {
        'N2': {'N'},
        'CO2': {'C', 'O'},
        'H2O': {'H', 'O'},
        'CO': {'C', 'O'},
        'H2': {'H'},
        'O2': {'O'},
        'NO': {'N', 'O'},
        'NO2': {'N', 'O'},
        'OH': {'O', 'H'},
        'CH4': {'C', 'H'},
        'NH3': {'N', 'H'},
        'HCN': {'H', 'C', 'N'},
        'HCl': {'H', 'Cl'},
        'Cl2': {'Cl'},
        'C_graphite': {'C'},
        'Al2O3': {'Al', 'O'},
        'AlO': {'Al', 'O'},
        'MgO': {'Mg', 'O'},
        'B2O3': {'B', 'O'},
    }
    
    mask = []
    for species in species_list:
        required_elements = species_elements.get(species, set())
        # 物种活性：所需元素都可用
        is_active = required_elements.issubset(available_elements)
        mask.append(1.0 if is_active else 0.0)
    
    return jnp.array(mask)


def generate_element_mask(
    formula_vector: jnp.ndarray,
    threshold: float = 1e-10
) -> jnp.ndarray:
    """根据元素含量生成掩码
    
    Args:
        formula_vector: 元素含量向量
        threshold: 最小含量阈值
        
    Returns:
        元素活性掩码
    """
    return jnp.where(formula_vector > threshold, 1.0, 0.0)


def apply_mask(
    values: jnp.ndarray,
    mask: jnp.ndarray,
    fill_value: float = 0.0
) -> jnp.ndarray:
    """应用掩码到数值数组
    
    Args:
        values: 原始数值
        mask: 掩码 (1.0/0.0)
        fill_value: 非活性位置的填充值
        
    Returns:
        掩码后的数组
    """
    return jnp.where(mask > 0.5, values, fill_value)


def get_default_species_list() -> List[str]:
    """获取默认产物物种列表"""
    return [
        'N2', 'CO2', 'H2O', 'CO', 'H2', 'O2',
        'NO', 'NO2', 'OH', 'CH4', 'NH3', 'HCN',
        'C_graphite', 'Al2O3', 'MgO', 'B2O3'
    ]


def get_elements_from_components(component_names: List[str]) -> Set[str]:
    """从组分名称获取所有元素
    
    Args:
        component_names: 组分名称列表
        
    Returns:
        元素集合
    """
    from pdu.data.components import get_component
    
    elements = set()
    for name in component_names:
        comp = get_component(name)
        elements.update(comp.formula.keys())
    
    return elements
