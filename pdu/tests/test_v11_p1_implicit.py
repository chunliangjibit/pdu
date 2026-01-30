# pdu/tests/test_v11_p1_implicit.py
import jax
import jax.numpy as jnp
from pdu.api import Recipe
from pdu.thermo.implicit_eos import get_balanced_thermo
from pdu.data.components import get_component
from pdu.data.products import load_products
from pdu.core.equilibrium import build_stoichiometry_matrix
import os
import json
from pathlib import Path

def test_hmx_balanced_state():
    # 1. 设置 HMX 环境
    comp = get_component('HMX')
    rho_loading = 1.89 # g/cm3
    e_internal = comp.heat_of_formation * 1000.0 / (comp.molecular_weight / 1000.0) # J/kg approx
    
    # 获取原子向量和系数
    SPECIES_LIST = ('N2', 'H2O', 'CO2', 'CO', 'C_graphite', 'H2', 'O2')
    ELEMENT_LIST = ('C', 'H', 'N', 'O', 'Al')
    A_matrix = build_stoichiometry_matrix(SPECIES_LIST, ELEMENT_LIST)
    atomic_masses = jnp.array([12.011, 1.008, 14.007, 15.999, 26.982])
    
    # 加载 JCZ3 参数
    data_dir = Path('pdu/data')
    with open(data_dir / 'jcz3_params.json') as f:
        v7_params = json.load(f)['species']
    
    eps, r, alpha, lam = [], [], [], []
    for s in SPECIES_LIST:
        p = v7_params[s]
        eps.append(p.get('epsilon_over_k', 100.0))
        r.append(p.get('r_star', 3.5))
        alpha.append(p.get('alpha', 13.0))
        lam.append(p.get('lambda_ree', 0.0))
    
    eos_params = (jnp.array(eps), jnp.array(r), jnp.array(alpha), jnp.array(lam),
                  jnp.zeros(len(SPECIES_LIST)), jnp.zeros(len(SPECIES_LIST)), # solid mask/v0
                  0.0, 10.0, 0.0, 0.0) # n_fixed, v0_fixed, e_fixed, r_corr
    
    # NASA 系数
    products_db = load_products()
    coeff_list = []
    for s in SPECIES_LIST:
        p = products_db[s]
        c7 = p.coeffs_high[:7]
        coeff_list.append(jnp.concatenate([jnp.zeros(2), c7]))
    coeffs_all = jnp.stack(coeff_list)
    
    # HMX 原子向量 (C4H8N8O8) - 每摩尔化合物
    atom_vec = jnp.array([4.0, 8.0, 8.0, 8.0, 0.0]) 
    
    # 2. 运行能量平衡计算
    # HMX 爆热约 6 MJ/kg
    target_e_j_kg = 5.8e6
    
    print(f"Testing HMX balanced state at rho={rho_loading} g/cm3, e={target_e_j_kg/1e6:.1f} MJ/kg...")
    res = get_balanced_thermo(
        jnp.float32(rho_loading), 
        jnp.float32(target_e_j_kg), 
        atom_vec, coeffs_all, A_matrix, atomic_masses, eos_params
    )
    
    print(f"Result: P={res.P/1e9:.2f} GPa, T={res.T:.0f} K")
    assert res.P > 0
    assert res.T > 300

    # 3. 梯度检查
    print("\nChecking Gradients...")
    def loss_fn(r, e):
        # 模拟对压力的优化
        state = get_balanced_thermo(r, e, atom_vec, coeffs_all, A_matrix, atomic_masses, eos_params)
        return jnp.sum(state.P)
    
    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    # 稍微抖动 rho 看看响应
    drho_de = grad_fn(jnp.float32(rho_loading), jnp.float32(target_e_j_kg))
    print(f"Gradient dP/drho: {drho_de[0]:.2e}")
    print(f"Gradient dP/de:   {drho_de[1]:.2e}")
    
    # 验证梯度方向是否符合物理：增加密度 -> 增加压力；增加内能 -> 增加压力
    assert drho_de[0] > 0
    assert drho_de[1] > 0
    print("Gradients verification: SUCCESS")

if __name__ == "__main__":
    test_hmx_balanced_state()
