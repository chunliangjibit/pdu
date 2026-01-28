"""
数据加载模块 - 组分数据

加载和管理高能材料组分热力学数据。
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List
import jax.numpy as jnp


@dataclass
class ComponentData:
    """组分数据结构"""
    name: str
    full_name: str
    formula: Dict[str, int]
    molecular_weight: float
    density: float  # g/cm³
    heat_of_formation: float  # kJ/mol
    oxygen_balance: float  # %
    category: str
    
    @property
    def internal_energy_of_formation(self) -> float:
        """计算标准生成内能 ΔfU° (kJ/mol)
        
        ΔfU° = ΔfH° - Δ(PV)
        Δ(PV) ≈ Δn_gas * RT
        
        Reaction: Elements -> Compound(s)
        Δn_gas = n_gas_products (0) - n_gas_reactants
        
        Standard states:
        H -> 1/2 H2(g)
        N -> 1/2 N2(g)
        O -> 1/2 O2(g)
        F -> 1/2 F2(g)
        Cl -> 1/2 Cl2(g)
        C -> C(graphite, s)
        B -> B(s)
        Al -> Al(s)
        """
        # 气体分子原子数
        gas_elements = {'H': 2, 'N': 2, 'O': 2, 'F': 2, 'Cl': 2}
        
        n_gas_reactants = 0.0
        for elem, atoms_per_mol in gas_elements.items():
            n_atom = self.formula.get(elem, 0)
            n_gas_reactants += n_atom / atoms_per_mol
            
        # 假设生成物是固/液态 (n_gas_products = 0)
        # TODO: 如果未来支持气态反应物输入，这里需调整
        delta_n_gas = 0.0 - n_gas_reactants
        
        R = 8.314462618e-3  # kJ/(mol·K)
        T = 298.15          # K
        
        work_term = delta_n_gas * R * T
        
        # ΔU = ΔH - Δ(PV)
        return self.heat_of_formation - work_term

    def to_atom_vector(self, elements: List[str] = None) -> jnp.ndarray:
        """转换为原子向量
        
        Args:
            elements: 元素顺序列表，默认 ['C', 'H', 'N', 'O', 'Cl', 'Al', 'Mg', 'B', 'K']
            
        Returns:
            原子数向量
        """
        if elements is None:
            elements = ['C', 'H', 'N', 'O', 'Cl', 'Al', 'Mg', 'B', 'K']
        
        vector = []
        for elem in elements:
            vector.append(float(self.formula.get(elem, 0)))
        return jnp.array(vector)


# 全局缓存
_COMPONENTS_CACHE: Optional[Dict[str, ComponentData]] = None
_DATA_DIR = Path(__file__).parent


def _get_data_dir() -> Path:
    """获取数据目录路径"""
    return _DATA_DIR


def load_components(reload: bool = False) -> Dict[str, ComponentData]:
    """加载组分数据库
    
    Args:
        reload: 是否强制重新加载
        
    Returns:
        组分名称到 ComponentData 的映射
    """
    global _COMPONENTS_CACHE
    
    if _COMPONENTS_CACHE is not None and not reload:
        return _COMPONENTS_CACHE
    
    data_path = _get_data_dir() / "reactants.json"
    
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    components = {}
    
    # 解析各类别
    for category_key in ['explosives', 'oxidizers', 'metals', 'binders']:
        if category_key not in raw_data:
            continue
            
        for name, data in raw_data[category_key].items():
            components[name] = ComponentData(
                name=name,
                full_name=data.get('name', name),
                formula=data['formula'],
                molecular_weight=data.get('molecular_weight', data.get('effective_Mw', data.get('molecular_weight_effective', 0.0))),
                density=data['density'],
                heat_of_formation=data.get('enthalpy_formation', 0.0),
                oxygen_balance=data.get('oxygen_balance', 0.0),
                category=data.get('type', 'other')
            )
    
    _COMPONENTS_CACHE = components
    return components


def get_component(name: str) -> ComponentData:
    """获取单个组分数据
    
    Args:
        name: 组分名称 (如 'RDX', 'HMX')
        
    Returns:
        ComponentData 对象
        
    Raises:
        KeyError: 如果组分不存在
    """
    components = load_components()
    if name not in components:
        raise KeyError(f"Unknown component: {name}. Available: {list(components.keys())}")
    return components[name]


def list_components() -> List[str]:
    """列出所有可用组分名称"""
    return list(load_components().keys())


def get_components_by_category(category: str) -> Dict[str, ComponentData]:
    """获取指定类别的所有组分
    
    Args:
        category: 类别名称 ('nitramine', 'nitroaromatic', 'metal', 等)
        
    Returns:
        符合条件的组分字典
    """
    components = load_components()
    return {
        name: comp for name, comp in components.items() 
        if comp.category == category
    }


def compute_mixture_atom_vector(
    component_names: List[str],
    mass_fractions: jnp.ndarray,
    elements: List[str] = None
) -> jnp.ndarray:
    """计算混合物的原子向量（基于质量分数）
    
    Args:
        component_names: 组分名称列表
        mass_fractions: 质量分数数组 (归一化)
        elements: 元素顺序列表
        
    Returns:
        混合物原子向量 (mol/g)
    """
    if elements is None:
        elements = ['C', 'H', 'N', 'O', 'Cl', 'Al', 'Mg', 'B', 'K']
    
    total_atoms = jnp.zeros(len(elements))
    
    for i, name in enumerate(component_names):
        comp = get_component(name)
        # 每克该组分中的原子数
        atoms_per_gram = comp.to_atom_vector(elements) / comp.molecular_weight
        total_atoms = total_atoms + mass_fractions[i] * atoms_per_gram
    
    return total_atoms  # mol/g
