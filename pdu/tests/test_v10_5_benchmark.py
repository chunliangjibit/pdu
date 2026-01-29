"""
PDU V10.5 Full Skill Benchmark
==============================

Skill: Generate PDU Consultation Whitepaper
Scope: Full 9-Explosive Suite
Settings: V10.5 (Thermal Lag + Relaxed Penalty)
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pdu.api import detonation_forward
from pathlib import Path
import numpy as np

def run_v10_5_full_skill():
    print("=" * 75)
    print("PyDetonation-Ultra V10.5 Full Benchmark (Skill Execution)")
    print("=" * 75)
    print()

    # Full 9-Explosive List from V10.4
    benchmarks = [
        {
            "name": "HMX",
            "recipe": {"HMX": 1.0},
            "rho0": 1.891,
            "exp": {"D": 9110, "P": 42.0, "T": 3800, "Q": 6.19},
            "jwl_ref": {"A": 778.0, "B": 7.07, "R1": 4.2, "R2": 1.0, "omega": 0.30}
        },
        {
            "name": "RDX",
            "recipe": {"RDX": 1.0},
            "rho0": 1.806,
            "exp": {"D": 8750, "P": 34.7, "T": 3600, "Q": 5.53},
            "jwl_ref": {"A": 778.0, "B": 7.07, "R1": 4.5, "R2": 1.4, "omega": 0.25}
        },
        {
            "name": "PETN",
            "recipe": {"PETN": 1.0},
            "rho0": 1.760,
            "exp": {"D": 8260, "P": 31.5, "T": 4503, "Q": 5.81},
            "jwl_ref": {"A": 617.0, "B": 16.9, "R1": 4.4, "R2": 1.2, "omega": 0.25}
        },
        {
            "name": "TNT",
            "recipe": {"TNT": 1.0},
            "rho0": 1.630,
            "exp": {"D": 6930, "P": 21.0, "T": 3100, "Q": 4.69},
            "jwl_ref": {"A": 374.0, "B": 3.23, "R1": 4.15, "R2": 0.90, "omega": 0.30}
        },
        {
            "name": "NM",
            "recipe": {"NM": 1.0},
            "rho0": 1.128,
            "exp": {"D": 6260, "P": 12.6, "T": 3600, "Q": 4.40},
            "jwl_ref": {"A": 488.0, "B": 6.36, "R1": 5.0, "R2": 1.5, "omega": 0.38}
        },
        {
            "name": "Comp B",
            "recipe": {"RDX": 0.60, "TNT": 0.40},
            "rho0": 1.717,
            "exp": {"D": 7980, "P": 29.5, "T": 3400, "Q": 5.10},
            "jwl_ref": {"A": 524.0, "B": 7.68, "R1": 4.2, "R2": 1.1, "omega": 0.34}
        },
        {
            "name": "Octol (75/25)",
            "recipe": {"HMX": 0.75, "TNT": 0.25},
            "rho0": 1.821,
            "exp": {"D": 8480, "P": 34.8, "T": 3500, "Q": 5.60},
            "jwl_ref": {"A": 648.0, "B": 12.95, "R1": 4.5, "R2": 1.4, "omega": 0.28}
        },
        {
            "name": "Tritonal (80/20)",
            "recipe": {"TNT": 0.80, "Al": 0.20},
            "rho0": 1.720,
            "exp": {"D": 6700, "P": 25.0, "T": 3200, "Q": 7.50},
            "jwl_ref": {"A": 400.0, "B": 4.0, "R1": 4.5, "R2": 1.2, "omega": 0.35}
        },
        {
            "name": "PBXN-109",
            "recipe": {"RDX": 0.64, "Al": 0.20, "HTPB_Cured": 0.16}, 
            "rho0": 1.68,
            "exp": {"D": 8050, "P": 30.0, "T": 3300, "Q": 9.7}, 
            "jwl_ref": {"A": 1157.0, "B": 19.4, "R1": 5.7, "R2": 1.242, "omega": 0.199}
        }
    ]

    results = []
    
    for b in benchmarks:
        print(f"Testing {b['name']}...")
        comps = list(b['recipe'].keys())
        fracs = list(b['recipe'].values())
        rho = b['rho0']
        
        is_aluminized = 'Al' in comps
        
        try:
            # V10.5: 使用 RELAXED_PENALTY 拟合算法
            res = detonation_forward(
                comps, fracs, rho, 
                verbose=False, # Reduce console noise
                reaction_degree=None, 
                combustion_efficiency=1.0,
                fitting_method='RELAXED_PENALTY'  # [P0-5]
            )
            
            def calc_err(pred, ref):
                if ref == 0: return 0.0
                return (pred - ref) / ref * 100.0

            err_D = calc_err(res.D, b['exp']['D'])
            err_P = calc_err(res.P_cj, b['exp']['P'])
            err_T = calc_err(res.T_cj, b['exp']['T'])
            err_Q = calc_err(res.Q, b['exp']['Q'])
            
            j_ref = b["jwl_ref"]
            miller_deg = getattr(res, 'miller_degree', 0.0)
            
            results.append({
                "name": b['name'],
                "D": [res.D, b['exp']['D'], err_D],
                "P": [res.P_cj, b['exp']['P'], err_P],
                "T": [res.T_cj, b['exp']['T'], err_T],
                "Q": [res.Q, b['exp']['Q'], err_Q],
                "is_aluminized": is_aluminized,
                "miller_degree": miller_deg,
                "JWL": {
                   "A": [res.jwl_A, j_ref['A'], calc_err(res.jwl_A, j_ref['A'])],
                   "B": [res.jwl_B, j_ref['B'], calc_err(res.jwl_B, j_ref['B'])],
                   "R1": [res.jwl_R1, j_ref['R1'], calc_err(res.jwl_R1, j_ref['R1'])],
                   "R2": [res.jwl_R2, j_ref['R2'], calc_err(res.jwl_R2, j_ref['R2'])],
                   "omega": [res.jwl_omega, j_ref['omega'], calc_err(res.jwl_omega, j_ref['omega'])]
                }
            })
            
            status = "🔥 Al" if is_aluminized else "  "
            b_val = res.jwl_B
            b_status = "✅" if b_val > 0.0 else "❌"
            print(f"  {status} D={res.D:.0f} P={res.P_cj:.1f} ({err_P:+.1f}%) | B={b_val:.2f} {b_status} | λ={miller_deg:.3f}")
            
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # 生成详细报告 (Skill Format)
    print()
    print("=" * 75)
    print("Generating Whitepaper Data...")
    print("=" * 75)
    
    report_path = Path('docs/v10_5_performance_report.md')
    with report_path.open('w', encoding='utf-8') as f:
        f.write("# PyDetonation-Ultra V10.5 Full Benchmark Report\n\n")
        f.write("Generated by Skill Execution\n\n")
        
        f.write("## 1. 爆轰性能汇总 (Predicted / Experimental [Error%])\n")
        f.write("| 序号 | 炸药 | D (m/s) | P (GPa) | T (K) | Q (MJ/kg) | λ_Al |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for i, r in enumerate(results):
            marker = "🔥 " if r['is_aluminized'] else ""
            f.write(f"| {i+1} | **{marker}{r['name']}** ")
            f.write(f"| {r['D'][0]:.0f} / {r['D'][1]:.0f} [{r['D'][2]:+.1f}%] ")
            f.write(f"| {r['P'][0]:.1f} / {r['P'][1]:.1f} [{r['P'][2]:+.1f}%] ")
            f.write(f"| {r['T'][0]:.0f} / {r['T'][1]:.0f} [{r['T'][2]:+.1f}%] ")
            f.write(f"| {r['Q'][0]:.2f} / {r['Q'][1]:.2f} [{r['Q'][2]:+.1f}%] ")
            f.write(f"| {r['miller_degree']:.3f} |\n")
            
        f.write("\n## 2. JWL 参数对标 (Relaxed Penalty Fit)\n")
        f.write("| 炸药 | A (GPa) | B (GPa) | R1 | R2 | ω |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            j = r['JWL']
            f.write(f"| **{r['name']}** ")
            
            def fmt_cell(item):
                pred, ref, err = item
                if ref == 0: return f"{pred:.2f} / 0.0"
                # If error is huge (>100%), just show values or shorten?
                # Skill request: Predicted / Experimental (Error%)
                return f"{pred:.1f} / {ref:.1f} [{err:+.1f}%]"

            f.write(f"| {fmt_cell(j['A'])} | {fmt_cell(j['B'])} | {fmt_cell(j['R1'])} | {fmt_cell(j['R2'])} | {fmt_cell(j['omega'])} |\n")
            
    print(f"\n📄 Report saved to {report_path}")

if __name__ == "__main__":
    run_v10_5_full_skill()
