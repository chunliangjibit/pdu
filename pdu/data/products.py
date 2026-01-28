"""
数据加载模块 - 产物数据

加载和管理爆轰产物 NASA 多项式数据。
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import jax.numpy as jnp


@dataclass 
class ProductData:
    """产物热力学数据结构"""
    name: str
    full_name: str
    molecular_weight: float  # g/mol
    phase: str  # 'gas', 'solid', 'liquid'
    coeffs_high: jnp.ndarray  # 高温多项式系数 (1000-5000 K)
    coeffs_low: jnp.ndarray   # 低温多项式系数 (300-1000 K)
    
    def get_coeffs(self, T: float) -> jnp.ndarray:
        """根据温度选择合适的多项式系数
        
        Args:
            T: 温度 (K)
            
        Returns:
            NASA 7系数多项式系数
        """
        return jnp.where(T > 1000.0, self.coeffs_high, self.coeffs_low)


# 全局缓存
_PRODUCTS_CACHE: Optional[Dict[str, ProductData]] = None
_JCZ3_CACHE: Optional[Dict] = None
_DATA_DIR = Path(__file__).parent.parent.parent / "data_raw"


def _get_data_dir() -> Path:
    """获取数据目录路径"""
    return _DATA_DIR


def load_products(reload: bool = False) -> Dict[str, ProductData]:
    """加载产物 NASA 多项式数据库
    
    Args:
        reload: 是否强制重新加载
        
    Returns:
        产物名称到 ProductData 的映射
    """
    global _PRODUCTS_CACHE
    
    if _PRODUCTS_CACHE is not None and not reload:
        return _PRODUCTS_CACHE
    
    data_path = _get_data_dir() / "products.json"
    
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    products = {}
    
    for name, data in raw_data.get('species', {}).items():
        products[name] = ProductData(
            name=name,
            full_name=data['name'],
            molecular_weight=data['molecular_weight'],
            phase=data['phase'],
            coeffs_high=jnp.array(data['coeffs_high']),
            coeffs_low=jnp.array(data['coeffs_low'])
        )
    
    _PRODUCTS_CACHE = products
    return products


def get_product_thermo(name: str) -> ProductData:
    """获取单个产物的热力学数据
    
    Args:
        name: 产物名称 (如 'N2', 'CO2')
        
    Returns:
        ProductData 对象
        
    Raises:
        KeyError: 如果产物不存在
    """
    products = load_products()
    if name not in products:
        raise KeyError(f"Unknown product: {name}. Available: {list(products.keys())}")
    return products[name]


def list_products() -> List[str]:
    """列出所有可用产物名称"""
    return list(load_products().keys())


def get_products_by_phase(phase: str) -> Dict[str, ProductData]:
    """获取指定相态的所有产物
    
    Args:
        phase: 相态 ('gas', 'solid', 'liquid')
        
    Returns:
        符合条件的产物字典
    """
    products = load_products()
    return {
        name: prod for name, prod in products.items() 
        if prod.phase == phase
    }


def load_jcz3_params(reload: bool = False) -> Dict:
    """加载 JCZ3 Exp-6 势能参数
    
    Args:
        reload: 是否强制重新加载
        
    Returns:
        JCZ3 参数字典
    """
    global _JCZ3_CACHE
    
    if _JCZ3_CACHE is not None and not reload:
        return _JCZ3_CACHE
    
    data_path = _get_data_dir() / "jcz3_params.json"
    
    with open(data_path, 'r', encoding='utf-8') as f:
        _JCZ3_CACHE = json.load(f)
    
    return _JCZ3_CACHE


def get_exp6_params(species: str) -> Tuple[float, float, float]:
    """获取物种的 Exp-6 势能参数
    
    Args:
        species: 物种名称
        
    Returns:
        (epsilon/k, r_star, alpha) 元组
        
    Raises:
        KeyError: 如果物种参数不存在
    """
    jcz3_data = load_jcz3_params()
    species_data = jcz3_data.get('species', {})
    
    if species not in species_data:
        raise KeyError(f"No Exp-6 parameters for species: {species}")
    
    params = species_data[species]
    return (
        params['epsilon_over_k'],
        params['r_star'],
        params['alpha']
    )


def get_all_species_params() -> Dict[str, Tuple[float, float, float]]:
    """获取所有物种的 Exp-6 参数
    
    Returns:
        物种名到 (epsilon/k, r_star, alpha) 的映射
    """
    jcz3_data = load_jcz3_params()
    species_data = jcz3_data.get('species', {})
    
    result = {}
    for name, params in species_data.items():
        result[name] = (
            params['epsilon_over_k'],
            params['r_star'],
            params['alpha']
        )
    return result
