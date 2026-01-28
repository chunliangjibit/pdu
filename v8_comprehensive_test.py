
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pdu.api import detonation_forward
import json
from pathlib import Path

def run_verification():
    print("PyDetonation-Ultra V8 Comprehensive Verification Suite")
    print("======================================================\n")

    # Experimental Benchmarks
    # Source: Web Search & Literature (LLNL, DTIC, EXPLO5)
    benchmarks = [
        {
            "name": "HMX",
            "recipe": {"HMX": 1.0},
            "rho0": 1.891,
            "exp_D": 9110,
            "exp_P": 42.0,
            "type": "Single"
        },
        {
            "name": "RDX",
            "recipe": {"RDX": 1.0},
            "rho0": 1.806,
            "exp_D": 8750,
            "exp_P": 34.7,
            "type": "Single"
        },
        {
            "name": "PETN",
            "recipe": {"PETN": 1.0},
            "rho0": 1.76,
            "exp_D": 8260,
            "exp_P": 31.5,
            "type": "Single"
        },
        {
            "name": "TNT",
            "recipe": {"TNT": 1.0},
            "rho0": 1.63,
            "exp_D": 6930,
            "exp_P": 21.0,
            "type": "Single"
        },
        {
            "name": "NM",
            "recipe": {"NM": 1.0},
            "rho0": 1.128,
            "exp_D": 6280,
            "exp_P": 12.5,
            "type": "Single"
        },
        {
            "name": "PBXN-109",
            "recipe": {"RDX": 0.64, "Al": 0.20, "HTPB_Cured": 0.16},
            "rho0": 1.68,
            "exp_D": 8050,
            "exp_P": 22.0, # Estimated High-Fidelity CJ
            "type": "Aluminized"
        },
        {
            "name": "Comp B",
            "recipe": {"RDX": 0.60, "TNT": 0.40},
            "rho0": 1.717,
            "exp_D": 7890,
            "exp_P": 26.5,
            "type": "Mixture"
        },
        {
            "name": "Octol (75/25)",
            "recipe": {"HMX": 0.75, "TNT": 0.25},
            "rho0": 1.81,
            "exp_D": 8480,
            "exp_P": 34.2,
            "type": "Mixture"
        },
        {
            "name": "Tritonal",
            "recipe": {"TNT": 0.80, "Al": 0.20},
            "rho0": 1.73,
            "exp_D": 6700,
            "exp_P": 18.5,
            "type": "Aluminized"
        }
    ]

    report_data = []

    for b in benchmarks:
        print(f"Calculating {b['name']} ({b['type']})...")
        comps = list(b['recipe'].keys())
        fracs = list(b['recipe'].values())
        
        try:
            res = detonation_forward(comps, fracs, b['rho0'], verbose=False)
            
            # Error calculation
            err_D = (res.D - b['exp_D']) / b['exp_D'] * 100.0
            err_P = (res.P_cj - b['exp_P']) / b['exp_P'] * 100.0
            
            data = {
                "name": b['name'],
                "type": b['type'],
                "rho0": b['rho0'],
                "exp_D": b['exp_D'],
                "pred_D": float(res.D),
                "err_D": float(err_D),
                "exp_P": b['exp_P'],
                "pred_P": float(res.P_cj),
                "err_P": float(err_P),
                "jwl": {
                    "A": float(res.jwl_A),
                    "B": float(res.jwl_B),
                    "R1": float(res.jwl_R1),
                    "R2": float(res.jwl_R2),
                    "omega": float(res.jwl_omega)
                }
            }
            report_data.append(data)
            print(f"  D: Exp={b['exp_D']}, Pred={res.D:.0f} ({err_D:+.2f}%)")
            print(f"  P: Exp={b['exp_P']}, Pred={res.P_cj:.2f} ({err_P:+.2f}%)\n")
            
        except Exception as e:
            print(f"  FAILED: {e}\n")

    # Generate Markdown Report
    report_path = Path('docs/v8_performance_report.md')
    with report_path.open('w', encoding='utf-8') as f:
        f.write("# PyDetonation-Ultra V8.4 性能验证报告 (最终物理修正版)\n\n")
        f.write("## 1. 爆轰基础性能 (D, P)\n\n")
        f.write("| 炸药名称 | 种类 | 装药密度 | 实验爆速 | 预测爆速 | 误差 | 实验爆压 | 预测爆压 | 误差 |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for d in report_data:
            f.write(f"| **{d['name']}** | {d['type']} | {d['rho0']} | {d['exp_D']} | {d['pred_D']:.0f} | {d['err_D']:+.2f}% | {d['exp_P']} | {d['pred_P']:.2f} | {d['err_P']:+.2f}% |\n")
        
        f.write("\n## 2. JWL 状态方程拟合参数\n\n")
        f.write("| 炸药名称 | A (GPa) | B (GPa) | R1 | R2 | omega | 拟合性质 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for d in report_data:
            j = d['jwl']
            f.write(f"| **{d['name']}** | {j['A']:.1f} | {j['B']:.2f} | {j['R1']:.2f} | {j['R2']:.2f} | {j['omega']:.4f} | 等熵锚定 |\n")

    # Generate JSON for archival
    with open('verification_results.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nReport generated: {report_path}")

if __name__ == "__main__":
    run_verification()
