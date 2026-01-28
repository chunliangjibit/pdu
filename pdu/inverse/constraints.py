"""
约束定义模块

配方优化的约束条件定义。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RecipeConstraints:
    """配方约束类
    
    定义优化配方时的各种约束条件。
    """
    
    # 组分质量分数约束
    min_fractions: Dict[str, float] = field(default_factory=dict)
    max_fractions: Dict[str, float] = field(default_factory=dict)
    
    # 必须/禁止组分
    required_components: List[str] = field(default_factory=list)
    forbidden_components: List[str] = field(default_factory=list)
    
    # 密度约束
    min_density: float = 1.0
    max_density: float = 2.2
    
    # 性能约束
    min_detonation_velocity: Optional[float] = None
    max_detonation_velocity: Optional[float] = None
    min_pressure: Optional[float] = None
    max_pressure: Optional[float] = None
    
    # 安全约束
    min_oxygen_balance: Optional[float] = None
    max_oxygen_balance: Optional[float] = None
    min_h50: Optional[float] = None  # 最小感度 (越大越钝感)
    
    def validate(self, component_names: List[str]) -> bool:
        """验证约束是否合法
        
        Args:
            component_names: 可用组分列表
            
        Returns:
            是否合法
        """
        # 检查必须组分是否在可用列表中
        for comp in self.required_components:
            if comp not in component_names:
                return False
        
        # 检查分数约束
        for comp in self.min_fractions:
            if comp not in component_names:
                return False
        
        for comp in self.max_fractions:
            if comp not in component_names:
                return False
        
        # 检查密度范围
        if self.min_density >= self.max_density:
            return False
        
        return True
    
    @classmethod
    def for_insensitive_explosive(cls) -> "RecipeConstraints":
        """钝感炸药约束预设"""
        return cls(
            min_density=1.7,
            max_density=2.0,
            min_h50=30.0,  # 至少 30cm
            max_oxygen_balance=0.0,
        )
    
    @classmethod
    def for_high_performance(cls) -> "RecipeConstraints":
        """高性能炸药约束预设"""
        return cls(
            min_density=1.8,
            max_density=2.1,
            min_detonation_velocity=8000.0,
            min_pressure=30.0,
        )
    
    @classmethod
    def for_propellant(cls) -> "RecipeConstraints":
        """推进剂约束预设"""
        return cls(
            min_density=1.5,
            max_density=1.9,
            max_oxygen_balance=5.0,
            min_oxygen_balance=-20.0,
        )


def check_constraints(
    recipe: Dict[str, float],
    density: float,
    performance: Dict[str, float],
    constraints: RecipeConstraints
) -> tuple[bool, List[str]]:
    """检查配方是否满足约束
    
    Args:
        recipe: 配方 (组分: 质量分数)
        density: 密度
        performance: 性能参数
        constraints: 约束条件
        
    Returns:
        (是否满足所有约束, 违反的约束列表)
    """
    violations = []
    
    # 密度约束
    if density < constraints.min_density:
        violations.append(f"密度 {density:.2f} < 最小值 {constraints.min_density}")
    if density > constraints.max_density:
        violations.append(f"密度 {density:.2f} > 最大值 {constraints.max_density}")
    
    # 必须组分
    for comp in constraints.required_components:
        if comp not in recipe or recipe[comp] < 0.01:
            violations.append(f"缺少必须组分 {comp}")
    
    # 禁止组分
    for comp in constraints.forbidden_components:
        if comp in recipe and recipe[comp] > 0.01:
            violations.append(f"包含禁止组分 {comp}")
    
    # 分数约束
    for comp, min_frac in constraints.min_fractions.items():
        if comp in recipe and recipe[comp] < min_frac:
            violations.append(f"{comp} 分数 {recipe[comp]:.2%} < {min_frac:.2%}")
    
    for comp, max_frac in constraints.max_fractions.items():
        if comp in recipe and recipe[comp] > max_frac:
            violations.append(f"{comp} 分数 {recipe[comp]:.2%} > {max_frac:.2%}")
    
    # 性能约束
    if constraints.min_detonation_velocity is not None:
        if performance.get('D', 0) < constraints.min_detonation_velocity:
            violations.append(f"爆速 {performance['D']:.0f} < {constraints.min_detonation_velocity:.0f}")
    
    if constraints.max_detonation_velocity is not None:
        if performance.get('D', float('inf')) > constraints.max_detonation_velocity:
            violations.append(f"爆速 {performance['D']:.0f} > {constraints.max_detonation_velocity:.0f}")
    
    # 安全约束
    if constraints.min_h50 is not None:
        if performance.get('h50', 0) < constraints.min_h50:
            violations.append(f"感度 h50={performance['h50']:.1f}cm < {constraints.min_h50:.1f}cm")
    
    return len(violations) == 0, violations
