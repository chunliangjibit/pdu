
import jax
jax.config.update("jax_enable_x64", True)
from pdu.api import detonation_forward
import numpy as np

def verify_compb():
    print("Comp B (RDX/TNT 64/36) V8.4 Performance Verification")
    print("-----------------------------------------------------")
    
    # Recipe and Density
    components = ["RDX", "TNT"]
    fractions = [0.64, 0.36]
    density = 1.717
    
    # Literature Benchmarks (LLNL / EXPLO5)
    exp_D = 7890
    exp_P = 26.8
    exp_T = 3400
    exp_A = 524.2
    exp_w = 0.34
    
    # Run forward calculation (Prior Blending is automatic)
    res = detonation_forward(components, fractions, density, verbose=True)
    
    print("\n[Comparison with Literature]")
    print(f"D (m/s):   {res.D:.1f} vs {exp_D} ({ (res.D - exp_D)/exp_D*100:+.2f}%)")
    print(f"P_cj (GPa): {res.P_cj:.2f} vs {exp_P} ({ (res.P_cj - exp_P)/exp_P*100:+.2f}%)")
    print(f"T_cj (K):   {res.T_cj:.0f} vs {exp_T} ({ (res.T_cj - exp_T)/exp_T*100:+.2f}%)")
    print(f"JWL A:      {res.jwl_A:.2f} vs {exp_A} ({ (res.jwl_A - exp_A)/exp_A*100:+.2f}%)")
    print(f"JWL omega:  {res.jwl_omega:.3f} vs {exp_w} ({ (res.jwl_omega - exp_w)/exp_w*100:+.2f}%)")
    
    # Check Curve MAE
    V = np.linspace(1.1, 5.0, 20)
    # Predicted curve
    def p_jwl(v, A, B, R1, R2): return A * np.exp(-R1 * v) + B * np.exp(-R2 * v)
    P_pred = p_jwl(V, res.jwl_A, res.jwl_B, res.jwl_R1, res.jwl_R2)
    # Exp curve proxy (using LLNL coefficients)
    # Comp B LLNL: A=524.2, B=7.67, R1=4.2, R2=1.1, w=0.34
    P_exp = p_jwl(V, 524.2, 7.67, 4.2, 1.1)
    mae = np.mean(np.abs(P_pred - P_exp))
    print(f"JWL Curve MAE: {mae:.3f} GPa")

if __name__ == "__main__":
    verify_compb()
