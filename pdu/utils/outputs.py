"""
输出格式化工具模块

提供结果格式化和报告生成功能。
"""

from typing import Dict, Any, List, Optional
import json


def format_results(
    result: Any,
    format_type: str = 'dict',
    precision: int = 4
) -> Any:
    """格式化计算结果
    
    Args:
        result: 计算结果对象
        format_type: 输出格式 ('dict', 'json', 'table')
        precision: 数值精度
        
    Returns:
        格式化后的结果
    """
    # 转换为字典
    if hasattr(result, '_asdict'):
        data = result._asdict()
    elif hasattr(result, '__dict__'):
        data = vars(result)
    else:
        data = dict(result) if isinstance(result, dict) else {'value': result}
    
    # 格式化数值
    def format_value(v):
        if isinstance(v, float):
            return round(v, precision)
        elif isinstance(v, dict):
            return {k: format_value(vv) for k, vv in v.items()}
        elif isinstance(v, (list, tuple)):
            return [format_value(vv) for vv in v]
        else:
            return v
    
    formatted = {k: format_value(v) for k, v in data.items()}
    
    if format_type == 'json':
        return json.dumps(formatted, indent=2, ensure_ascii=False)
    elif format_type == 'table':
        return _format_as_table(formatted)
    else:
        return formatted


def _format_as_table(data: Dict) -> str:
    """将字典格式化为表格字符串"""
    lines = []
    max_key_len = max(len(str(k)) for k in data.keys())
    
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for k, v in value.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"{key.ljust(max_key_len)}: {value}")
    
    return '\n'.join(lines)


def generate_report(
    recipe: Dict[str, float],
    performance: Dict[str, float],
    title: str = "爆轰计算报告"
) -> str:
    """生成计算报告
    
    Args:
        recipe: 配方字典
        performance: 性能参数字典
        title: 报告标题
        
    Returns:
        报告字符串
    """
    lines = [
        "=" * 50,
        f" {title}",
        "=" * 50,
        "",
        "【配方信息】",
    ]
    
    for comp, frac in recipe.items():
        lines.append(f"  {comp}: {frac*100:.1f}%")
    
    lines.extend([
        "",
        "【爆轰性能】",
    ])
    
    perf_labels = {
        'D': ('爆速', 'm/s', 0),
        'P_cj': ('CJ压力', 'GPa', 2),
        'T_cj': ('CJ温度', 'K', 0),
        'rho_cj': ('CJ密度', 'g/cm³', 3),
        'Q': ('爆热', 'kJ/kg', 0),
        'OB': ('氧平衡', '%', 1),
        'h50': ('撞击感度', 'cm', 1),
    }
    
    for key, value in performance.items():
        if key in perf_labels:
            label, unit, prec = perf_labels[key]
            lines.append(f"  {label}: {value:.{prec}f} {unit}")
    
    lines.extend([
        "",
        "=" * 50,
    ])
    
    return '\n'.join(lines)


def print_comparison(
    predicted: Dict[str, float],
    experimental: Dict[str, float],
    labels: Optional[Dict[str, str]] = None
) -> str:
    """打印预测值与实验值对比
    
    Args:
        predicted: 预测值字典
        experimental: 实验值字典
        labels: 参数标签
        
    Returns:
        对比表格字符串
    """
    if labels is None:
        labels = {k: k for k in predicted.keys()}
    
    lines = [
        "-" * 50,
        f"{'参数':<15} {'预测值':<12} {'实验值':<12} {'误差%':<10}",
        "-" * 50,
    ]
    
    for key in predicted.keys():
        if key in experimental:
            pred = predicted[key]
            exp = experimental[key]
            error = (pred - exp) / exp * 100 if exp != 0 else 0
            label = labels.get(key, key)
            lines.append(f"{label:<15} {pred:<12.2f} {exp:<12.2f} {error:<10.2f}")
    
    lines.append("-" * 50)
    
    return '\n'.join(lines)
