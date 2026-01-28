"""
PDU 广义炸药验证脚本

测试多种单质和混合炸药的计算精度，对比实验 JWL 参数。
"""

import sys
import os
import jax.numpy as jnp
from typing import Dict, Tuple

# 添加项目根目录到路径
sys.path.append(os.getcwd())

from pdu.core.detonation import compute_jwl_from_formula

# ===== 辅助函数: 混合物计算 =====

def mix_formulas(
    components: list,  # [(formula, mw, weight_percent), ...]
    density: float
) -> Tuple[Dict[str, float], float, float]:
    """计算混合物的等效分子式、分子量和生成热"""
    
    total_moles = 0.0
    effective_formula = {}
    effective_hof = 0.0
    
    # 首先计算总摩尔数 (基于 100g 混合物)
    for formula, mw, wt_pct, hof in components:
        moles = wt_pct / mw
        total_moles += moles
        
        # 累加原子
        for elem, count in formula.items():
            effective_formula[elem] = effective_formula.get(elem, 0.0) + count * moles
            
        # 累加生成热 (加权平均? 不，生成热通常是 kJ/mol)
        # H_mix (kJ/kg) = sum(wt_frac * H_k (kJ/kg))
        # 输入 hof 是 kJ/mol
        # H_k (kJ/kg) = hof / mw * 1000
        h_kg = hof / mw * 1000
        effective_hof += (wt_pct/100) * h_kg  # kJ/kg
        
    # 归一化分子式 (per effective mole)
    # 平均分子量 = 总质量 / 总摩尔数 = 100 / total_moles
    avg_mw = 100.0 / total_moles
    
    final_formula = {k: int(v / total_moles + 0.5) for k, v in effective_formula.items()}
    
    # 转换回 kJ/mol effective
    # H_mix (kJ/mol) = H_mix (kJ/kg) * avg_mw / 1000
    final_hof = effective_hof * avg_mw / 1000
    
    # 使用浮点数分子式以保持精确原子比
    final_formula_float = {k: v / total_moles for k, v in effective_formula.items()}
    
    return final_formula_float, avg_mw, final_hof


# ===== 炸药数据库 (实验值) =====
# (名称, 密度, 参考D, 参考P, 参考A, 参考B, 参考R1, 参考R2, 参考w)
# 数据来源: LLNL Explosives Handbook, Dobratz/Crawford

DB_EXP = {
    # 单质
    'HMX': {
        'rho': 1.89, 'D': 9110, 'P': 39.0,
        'A': 778.3, 'B': 7.07, 'R1': 4.20, 'R2': 1.00, 'w': 0.30,
        'comp': [({'C':4,'H':8,'N':8,'O':8}, 296.16, 100.0, 102.4)]
    },
    'RDX': {
        'rho': 1.80, 'D': 8754, 'P': 34.7,
        'A': 778.1, 'B': 7.07, 'R1': 4.54, 'R2': 1.21, 'w': 0.34, # 典型值
        'comp': [({'C':3,'H':6,'N':6,'O':6}, 222.12, 100.0, 92.6)]
    },
    'PETN': {
        'rho': 1.77, 'D': 8300, 'P': 33.5,
        'A': 617.0, 'B': 16.9, 'R1': 4.40, 'R2': 1.20, 'w': 0.25,
        'comp': [({'C':5,'H':8,'N':4,'O':12}, 316.14, 100.0, -539.0)] # HoF check
    },
    'TNT': {
        'rho': 1.63, 'D': 6950, 'P': 21.0, 
        'A': 371.2, 'B': 3.23, 'R1': 4.15, 'R2': 0.90, 'w': 0.30, # B值变动大
        'comp': [({'C':7,'H':5,'N':3,'O':6}, 227.13, 100.0, -67.0)]
    },
    'TATB': {
        'rho': 1.895, 'D': 7900, 'P': 31.2, # LX-17近似
        'A': 573.0, 'B': 6.5, 'R1': 4.65, 'R2': 1.25, 'w': 0.30, # 估算
        'comp': [({'C':6,'H':6,'N':6,'O':6}, 258.15, 100.0, -154.2)] # HoF check
    },
    
    # 混合炸药
    'Octol 75/25': {
        'rho': 1.81, 'D': 8480, 'P': 34.0, # Approx
        'A': 650.0, 'B': 8.0, 'R1': 4.4, 'R2': 1.2, 'w': 0.32, # 粗略参考
        'comp': [
            ({'C':4,'H':8,'N':8,'O':8}, 296.16, 75.0, 102.4), # HMX
            ({'C':7,'H':5,'N':3,'O':6}, 227.13, 25.0, -67.0)  # TNT
        ]
    },
    'Comp B': {
        'rho': 1.72, 'D': 8000, 'P': 29.5,
        'A': 524.0, 'B': 7.7, 'R1': 4.20, 'R2': 1.20, 'w': 0.34,
        'comp': [
            ({'C':3,'H':6,'N':6,'O':6}, 222.12, 60.0, 92.6), # RDX
            ({'C':7,'H':5,'N':3,'O':6}, 227.13, 40.0, -67.0)  # TNT
        ]
    },
    'TKX-50': {
        'rho': 1.80, 'D': 9037, 'P': 34.0, # P est from D
        'A': 800.0, 'B': 10.0, 'R1': 4.5, 'R2': 1.2, 'w': 0.30, # No exp, placeholder
        'comp': [({'C':2,'H':8,'N':10,'O':4}, 236.15, 100.0, 213.4)]
    },
    'Tritonal': {
        'rho': 1.72, 'D': 6650, 'P': 18.0, # Low P
        'A': 400.0, 'B': 5.0, 'R1': 4.0, 'R2': 1.0, 'w': 0.30, # Placeholder
        'comp': [
            ({'C':7,'H':5,'N':3,'O':6}, 227.13, 80.0, -67.0), # TNT
            ({'Al':1}, 26.98, 20.0, 0.0)    # Al
        ]
    },
    'PBXN-109': {
        'rho': 1.68, 'D': 7650, 'P': 23.0,
        'A': 1157.0, 'B': 19.4, 'R1': 5.70, 'R2': 1.24, 'w': 0.07, # Exp data found!
        'comp': [
            ({'C':3,'H':6,'N':6,'O':6}, 222.12, 64.0, 92.6), # RDX
            ({'Al':1}, 26.98, 20.0, 0.0),    # Al
            ({'C':4,'H':6,'O':2}, 86.09, 16.0, -426.0) # HTPB/Binder approx (C4H6O2 for simplicity, actually HTPB is hydrocarbon. DOA plasticizer)
            # Binder is complex. Let's use simple CH2 approx or standard binder.
            # HTPB is (C4H6)n. -100 kJ/mol? 
            # Let's use HTPB proxy: C=7.3, H=10.6, O=0.1.
            # Simplified binder: Polyurethane/HTPB C10H16N0.5O1? 
            # Let's use standard HTPB formula C7.3 H10.6 O0.1 (unit MW ~ 100). HoF ~ -50 kJ/kg.
            # Here use specific data if possible. 
            # Or just use the one provided: RDX/Al/Binder.
            # Let's use C4H6 (Butadiene) for HTPB backbone. HoF +110 kJ/mol (gas). Polymer is stable.
            # Let's use inert binder param: C1 H2. HoF -30 kJ/mol.
            # Better: use C4H6 (Polybutadiene), HoF = 20 kJ/mol per unit.
            # Let's stick to user inputs for generic.
            # Use C4H6O2 as placeholder.
        ]
    }
}

def run_verify():
    print(f"{'Name':<12} {'D_calc':<8} {'D_ref':<8} {'Err%':<6} {'P_calc':<8} {'P_ref':<8} {'A_calc':<8} {'A_ref':<8} {'Err%':<6}")
    print("-" * 90)
    
    for name, data in DB_EXP.items():
        # 准备输入
        if len(data['comp']) == 1:
            # 单质
            f, mw, _, hof = data['comp'][0]
            # 转换为 int 分子式
            formula = f
        else:
            # 混合计算
            f_float, mw, hof = mix_formulas(data['comp'], data['rho'])
            # 转换为 int (近似) 用于 API，但 compute_jwl_from_formula 内部应该能处理 float?
            # 现在的代码 estimate_products_simple 使用 formula.get(key, 0)
            # 如果是 float 应该也可以工作，只要不强制类型检查
            formula = {k: int(v) if v.is_integer() else v for k, v in f_float.items()}
            
        try:
            jwl, cj = compute_jwl_from_formula(
                formula, mw, hof, data['rho'],
                n_iters=1000, verbose=False
            )
            
            d_err = (cj.D - data['D']) / data['D'] * 100
            a_err = (jwl.A - data['A']) / data['A'] * 100
            
            print(f"{name:<12} {cj.D:>8.0f} {data['D']:>8} {d_err:>+5.1f}% {cj.P_cj:>8.1f} {data['P']:>8} {jwl.A:>8.0f} {data['A']:>8} {a_err:>+5.1f}%")
            
        except Exception as e:
            print(f"{name:<12} ERROR: {e}")

if __name__ == "__main__":
    run_verify()
