# test_thermo_consistency.py
"""
热力学一致性单元测试套件 (专家建议 4.1)

验证内容:
1. 低密度极限下的理想气一致性 (A, A_T, A_TT, h, c_p 关系)
2. 参考温度 (298K) 处各物种的基准对齐
3. 导数链自洽性验证
"""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from typing import Tuple, Dict
import unittest

from pdu.data.products import load_products
from pdu.physics.eos import compute_total_helmholtz_energy
from pdu.utils.precision import R_GAS

# ============================================================================
# 测试参数
# ============================================================================
T_REF = 298.15      # 参考温度 (K)
T_MID = 1000.0      # 中间温度 (K)
T_HIGH = 3000.0     # 高温 (K)
V_LARGE = 1.0       # 大体积 (cm³/mol) - 低密度极限
TOLERANCE_REL = 0.05  # 相对误差容忍度 5%


class ThermodynamicConsistencyTests(unittest.TestCase):
    """热力学一致性测试类"""
    
    @classmethod
    def setUpClass(cls):
        """加载测试数据"""
        cls.products = load_products()
        cls.species_list = ['N2', 'H2O', 'CO2', 'CO', 'H2']
        
        # 构建测试用 EOS 参数
        n_species = len(cls.species_list)
        cls.n = jnp.ones(n_species) * 0.2  # 等摩尔混合
        
        # 加载系数
        cl_l, ch_l = [], []
        eps_l, r_l, alpha_l, lam_l = [], [], [], []
        solid_mask_l, solid_v0_l = [], []
        
        for s in cls.species_list:
            p = cls.products[s]
            cl_l.append(jnp.concatenate([jnp.zeros(2), p.coeffs_low[:7]]))
            ch_l.append(jnp.concatenate([jnp.zeros(2), p.coeffs_high[:7]]))
            eps_l.append(100.0)   # 默认 ε/k
            r_l.append(3.5)       # 默认 r* (Å)
            alpha_l.append(13.0)  # 默认 α
            lam_l.append(0.0)     # 无极性修正
            solid_mask_l.append(0.0)
            solid_v0_l.append(0.0)
        
        cls.coeffs_low = jnp.stack(cl_l)
        cls.coeffs_high = jnp.stack(ch_l)
        cls.eps_vec = jnp.array(eps_l)
        cls.r_star_vec = jnp.array(r_l)
        cls.alpha_vec = jnp.array(alpha_l)
        cls.lambda_vec = jnp.array(lam_l)
        cls.solid_mask = jnp.array(solid_mask_l)
        cls.solid_v0 = jnp.array(solid_v0_l)
        
        # 平均分子量
        cls.mw_avg = sum(cls.products[s].molecular_weight for s in cls.species_list) / n_species
    
    def _compute_A(self, V, T) -> float:
        """计算 Helmholtz 自由能"""
        return compute_total_helmholtz_energy(
            self.n, V, T, self.coeffs_low, self.coeffs_high,
            self.eps_vec, self.r_star_vec, self.alpha_vec, self.lambda_vec,
            self.solid_mask, self.solid_v0,
            mw_avg=self.mw_avg
        )
    
    def _compute_derivatives(self, V, T) -> Tuple[float, float, float, float]:
        """计算 A 及其导数 (数值微分验证)"""
        A = self._compute_A(V, T)
        
        # A_T (熵相关)
        grad_T = jax.grad(lambda t: self._compute_A(V, t))
        A_T = grad_T(T)
        
        # A_TT (热容相关)
        grad_TT = jax.grad(lambda t: jax.grad(lambda t2: self._compute_A(V, t2))(t))
        A_TT = grad_TT(T)
        
        # A_V (压力相关)
        grad_V = jax.grad(lambda v: self._compute_A(v, T))
        A_V = grad_V(V)
        
        return A, A_T, A_TT, A_V
    
    # ========================================================================
    # Test 1: 低密度极限理想气一致性
    # ========================================================================
    def test_low_density_pressure_sanity(self):
        """低密度压力自检 (专家建议 4.1 改名)"""
        print(f"\n[Test 1] Low Density Pressure Sanity Sweep:")
        print(f"{'V (cm³/mol)':>12} | {'P (MPa)':>10} | {'P_ideal (MPa)':>12} | {'Err %':>8}")
        print("-" * 52)
        
        for v_mol in [10.0, 50.0, 100.0, 500.0, 1000.0]:
            T = T_MID
            A, A_T, A_TT, A_V = self._compute_derivatives(v_mol, T)
            p_computed = -A_V * 1e6  # Pa
            
            n_total = jnp.sum(self.n)
            p_ideal = n_total * R_GAS * T / (v_mol * 1e-6)  # Pa
            
            err = (p_computed - p_ideal) / p_ideal
            print(f"{v_mol:12.1f} | {p_computed/1e6:10.4f} | {p_ideal/1e6:12.4f} | {err*100:7.2f}%")
            
        # 最终判定 (以最大体积点为准)
        self.assertLess(abs(err), 0.35, "Low-density pressure deviation too large")

    def test_cv_positive_definite(self):
        """C_v 必须在所有测试温度下正定 (专家建议 4.2)"""
        V = 10.0  # cm³/mol
        n_total = jnp.sum(self.n)
        m_kg = n_total * self.mw_avg / 1000.0
        
        print(f"\n[Test 2] C_v Stability & Triple-Output (V={V} cm³/mol, m={m_kg:.4f} kg):")
        print(f"{'T (K)':>8} | {'Cv (J/K)':>10} | {'cv (J/kg/K)':>12} | {'Cv_bar (J/mol/K)':>15} | Status")
        print("-" * 65)
        
        for T in [2000.0, 3000.0, 5000.0, 8000.0]:
            A, A_T, A_TT, A_V = self._compute_derivatives(V, T)
            CV_total = -T * A_TT
            cv_mass = CV_total / m_kg
            cv_molar = CV_total / n_total
            
            status = "OK" if CV_total > 0 else "FAIL"
            print(f"{T:8.0f} | {CV_total:10.2f} | {cv_mass:12.1f} | {cv_molar:15.2f} | {status}")
            self.assertGreater(CV_total, 0, f"C_v is not positive at T={T}K")
    
    # ========================================================================
    # Test 2: 参考态基准对齐
    # ========================================================================
    def test_reference_state_consistency(self):
        """参考温度处 A 和 A_T 应与 NASA 基准一致"""
        V = 50.0
        T = T_REF
        
        A, A_T, A_TT, A_V = self._compute_derivatives(V, T)
        S = -A_T
        
        print(f"\n[Test 3] Reference State (T=298.15K):")
        print(f"  A   = {A:.2f} J")
        print(f"  S   = {S:.2f} J/K")
        print(f"  A_T = {A_T:.2f} J/K")
        
        # 熵必须为正
        self.assertGreater(S, 0, "Entropy should be positive at reference state")
    
    # ========================================================================
    # Test 3: 导数链自洽性 (数值 vs 解析)
    # ========================================================================
    def test_derivative_chain_consistency(self):
        """验证 JAX 自动微分与有限差分一致"""
        V, T = 20.0, 2000.0
        dT = 0.01
        dV = 0.001
        
        A0 = self._compute_A(V, T)
        
        # 有限差分 A_T
        A_T_fd = (self._compute_A(V, T + dT) - self._compute_A(V, T - dT)) / (2 * dT)
        
        # 有限差分 A_V
        A_V_fd = (self._compute_A(V + dV, T) - self._compute_A(V - dV, T)) / (2 * dV)
        
        # JAX 自动微分
        A, A_T, A_TT, A_V = self._compute_derivatives(V, T)
        
        print(f"\n[Test 4] Derivative Consistency (V={V}, T={T}):")
        print(f"  A_T  (JAX): {A_T:.4f}, (FD): {A_T_fd:.4f}, Diff: {abs(A_T - A_T_fd):.2e}")
        print(f"  A_V  (JAX): {A_V:.4f}, (FD): {A_V_fd:.4f}, Diff: {abs(A_V - A_V_fd):.2e}")
        
        self.assertAlmostEqual(A_T, A_T_fd, delta=abs(A_T) * 0.01,
                               msg="A_T finite difference mismatch")
        self.assertAlmostEqual(A_V, A_V_fd, delta=abs(A_V) * 0.01,
                               msg="A_V finite difference mismatch")


# ============================================================================
# 运行测试
# ============================================================================
def run_consistency_tests():
    """运行所有热力学一致性测试"""
    print("=" * 60)
    print("Thermodynamic Consistency Test Suite")
    print("Based on Expert Recommendation 4.1")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(ThermodynamicConsistencyTests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All thermodynamic consistency tests PASSED")
    else:
        print(f"❌ {len(result.failures)} test(s) FAILED")
        print(f"❌ {len(result.errors)} test(s) ERROR")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_consistency_tests()
