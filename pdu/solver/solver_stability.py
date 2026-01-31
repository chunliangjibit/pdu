# solver_stability.py
"""
CJ 求解器可行域检测与稳定化框架 (专家建议 4.3)

功能:
1. 可行域约束 (V, T 硬边界)
2. 续接/同伦法 (λ 路径)
3. 发散日志记录
4. 分支判别
"""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, Optional, List, Callable
import numpy as np

# ============================================================================
# 可行域定义
# ============================================================================
@dataclass
class FeasibleDomain:
    """可行域约束"""
    V_min: float = 0.01     # cm³/mol, 最小体积 (防止高压发散)
    V_max: float = 100.0    # cm³/mol, 最大体积
    T_min: float = 200.0    # K, 最低温度
    T_max: float = 15000.0  # K, 最高温度
    P_max: float = 1000e9   # Pa, 最大压力阈值
    
    def check(self, V: float, T: float, P: float = None) -> Tuple[bool, str]:
        """检查状态是否在可行域内"""
        if V < self.V_min:
            return False, f"V={V:.4f} < V_min={self.V_min}"
        if V > self.V_max:
            return False, f"V={V:.4f} > V_max={self.V_max}"
        if T < self.T_min:
            return False, f"T={T:.0f} < T_min={self.T_min}"
        if T > self.T_max:
            return False, f"T={T:.0f} > T_max={self.T_max}"
        if P is not None and P > self.P_max:
            return False, f"P={P/1e9:.1f} GPa > P_max={self.P_max/1e9:.0f}"
        return True, "OK"
    
    def clamp(self, V: float, T: float) -> Tuple[float, float]:
        """将状态钳制到可行域内"""
        V_clamped = jnp.clip(V, self.V_min, self.V_max)
        T_clamped = jnp.clip(T, self.T_min, self.T_max)
        return V_clamped, T_clamped


# ============================================================================
# 发散日志
# ============================================================================
@dataclass
class DivergenceLog:
    """发散迭代日志 (专家建议 5.2)"""
    step: int
    V: float
    T: float
    P: float
    residual: float
    message: str


class SolverLogger:
    """求解器日志记录器"""
    
    def __init__(self):
        self.history: List[DivergenceLog] = []
        self.domain = FeasibleDomain()
    
    def log_step(self, step: int, V: float, T: float, P: float, residual: float):
        """记录一个迭代步"""
        is_feasible, msg = self.domain.check(V, T, P)
        
        log_entry = DivergenceLog(
            step=step,
            V=float(V),
            T=float(T),
            P=float(P),
            residual=float(residual),
            message=msg if not is_feasible else ""
        )
        self.history.append(log_entry)
        
        return is_feasible
    
    def get_divergence_report(self) -> str:
        """生成发散报告"""
        if not self.history:
            return "No iteration history recorded."
        
        report = []
        report.append("=" * 70)
        report.append("Solver Divergence Report (Expert Recommendation 5.2)")
        report.append("=" * 70)
        report.append(f"{'Step':>5} | {'V (cm³)':>10} | {'T (K)':>8} | {'P (GPa)':>10} | {'Residual':>12} | Status")
        report.append("-" * 70)
        
        for log in self.history[-20:]:  # 最后 20 步
            status = "INFEASIBLE" if log.message else "OK"
            report.append(
                f"{log.step:5d} | {log.V:10.4f} | {log.T:8.0f} | {log.P/1e9:10.2f} | {log.residual:12.2e} | {status}"
            )
        
        report.append("-" * 70)
        
        # 分析发散特征
        if self.history:
            last = self.history[-1]
            if last.P > 100e9:
                report.append("\n⚠️  DIAGNOSIS: High pressure branch detected.")
                report.append("   Likely cause: Newton step pushed V too small.")
            if last.T < 500:
                report.append("\n⚠️  DIAGNOSIS: Low temperature branch detected.")
                report.append("   Likely cause: Endotherm overcooling or no solution.")
            if last.V < 0.1:
                report.append("\n⚠️  DIAGNOSIS: Over-compression detected.")
                report.append("   Likely cause: Hugoniot-Rayleigh no intersection in feasible domain.")
        
        return "\n".join(report)
    
    def check_no_intersection(self) -> bool:
        """检查是否有"无交点"特征 (残差符号不变)"""
        if len(self.history) < 5:
            return False
        
        residuals = [log.residual for log in self.history[-5:]]
        signs = [r > 0 for r in residuals]
        
        # 如果最后 5 步残差符号全部相同，可能无交点
        return all(signs) or not any(signs)


# ============================================================================
# 同伦/续接法
# ============================================================================
def homotopy_solve(
    solve_func: Callable,
    initial_guess: Tuple[float, float],
    lambda_steps: int = 10,
    domain: FeasibleDomain = None
) -> Tuple[Optional[Tuple[float, float]], List[Tuple[float, float, float]]]:
    """
    同伦法求解 (专家建议 4.3)
    
    通过 λ ∈ [0, 1] 的路径逐步打开强耦合，每一步用上一解做初值。
    
    Args:
        solve_func: 求解函数，签名 solve_func(lambda, guess) -> (V, T, residual)
        initial_guess: 初始猜测 (V0, T0)
        lambda_steps: λ 路径步数
        domain: 可行域约束
    
    Returns:
        final_solution: (V, T) 或 None
        path: λ 路径上的 (λ, V, T) 列表
    """
    if domain is None:
        domain = FeasibleDomain()
    
    path = []
    current_guess = initial_guess
    
    print(f"\n[Homotopy Solve] Starting with {lambda_steps} steps...")
    print(f"{'λ':>6} | {'V':>10} | {'T':>8} | {'Residual':>12} | Status")
    print("-" * 55)
    
    for i in range(lambda_steps + 1):
        lam = i / lambda_steps
        
        try:
            V, T, residual = solve_func(lam, current_guess)
            
            is_feasible, msg = domain.check(V, T)
            status = "OK" if is_feasible else f"INFEAS: {msg}"
            
            print(f"{lam:6.2f} | {V:10.4f} | {T:8.0f} | {residual:12.2e} | {status}")
            
            if not is_feasible:
                print(f"\n⚠️  Homotopy stopped at λ={lam}: left feasible domain")
                return None, path
            
            path.append((lam, V, T))
            current_guess = (V, T)
            
        except Exception as e:
            print(f"\n❌ Homotopy failed at λ={lam}: {e}")
            return None, path
    
    print("-" * 55)
    print(f"✅ Homotopy completed: V={current_guess[0]:.4f}, T={current_guess[1]:.0f}")
    
    return current_guess, path


# ============================================================================
# 多初值扫描
# ============================================================================
def multi_initial_scan(
    solve_func: Callable,
    V_range: Tuple[float, float] = (5.0, 50.0),
    T_range: Tuple[float, float] = (2000.0, 8000.0),
    n_samples: int = 9
) -> List[Tuple[float, float, float, float]]:
    """
    多初值扫描 (专家建议 4.3)
    
    用多组初值检查是否存在多解/分支。
    
    Returns:
        List of (V_init, T_init, V_final, T_final) tuples
    """
    import numpy as np
    
    V_samples = np.linspace(V_range[0], V_range[1], int(np.sqrt(n_samples)))
    T_samples = np.linspace(T_range[0], T_range[1], int(np.sqrt(n_samples)))
    
    results = []
    
    print(f"\n[Multi-Initial Scan] Testing {len(V_samples) * len(T_samples)} initial points...")
    
    for V_init in V_samples:
        for T_init in T_samples:
            try:
                V_final, T_final, residual = solve_func((V_init, T_init))
                results.append((V_init, T_init, V_final, T_final, residual))
            except:
                results.append((V_init, T_init, None, None, None))
    
    # 分析是否有多分支
    valid_results = [r for r in results if r[2] is not None]
    if valid_results:
        V_finals = [r[2] for r in valid_results]
        T_finals = [r[3] for r in valid_results]
        
        V_std = np.std(V_finals)
        T_std = np.std(T_finals)
        
        print(f"\nResults Summary:")
        print(f"  Converged: {len(valid_results)}/{len(results)}")
        print(f"  V_final std: {V_std:.4f}")
        print(f"  T_final std: {T_std:.0f}")
        
        if V_std > 1.0 or T_std > 500:
            print(f"\n⚠️  WARNING: Large variance suggests multiple branches!")
        else:
            print(f"\n✅ Single solution branch detected.")
    
    return results


# ============================================================================
# 示例用法
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Solver Stability Framework")
    print("(Expert Recommendation 4.3)")
    print("=" * 60)
    
    # 示例: 可行域检查
    domain = FeasibleDomain()
    
    test_cases = [
        (10.0, 3000.0, 50e9),   # 正常
        (0.001, 3000.0, 50e9),  # V 太小
        (10.0, 100.0, 50e9),    # T 太低
        (10.0, 3000.0, 2000e9), # P 太高
    ]
    
    print("\n[Feasible Domain Check Examples]:")
    for V, T, P in test_cases:
        is_ok, msg = domain.check(V, T, P)
        status = "✅ OK" if is_ok else f"❌ {msg}"
        print(f"  V={V:6.3f}, T={T:5.0f}, P={P/1e9:6.0f} GPa -> {status}")
    
    # 示例: 发散日志
    print("\n[Divergence Logger Demo]:")
    logger = SolverLogger()
    
    # 模拟一个发散过程
    for i, (V, T, P) in enumerate([
        (10.0, 3000.0, 40e9),
        (5.0, 4000.0, 80e9),
        (1.0, 5000.0, 200e9),
        (0.1, 6000.0, 1000e9),
        (0.01, 300.0, 5000e9),
    ]):
        logger.log_step(i, V, T, P, 1e10 / (i + 1))
    
    print(logger.get_divergence_report())
