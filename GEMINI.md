# PyDetonation-Ultra (PDU) Project Context

## **CRITICAL PROTOCOL: SKILL-FIRST WORKFLOW (MANDATORY)**
**RULE #1:** Whenever you encounter a problem, need to implement a feature, or plan a task, you **MUST FIRST** search the `/home/jcl/HDY/PyDetonation-Ultra/.agent/skills` directory.
- **Action**: Use `find_by_name` or `grep_search` to find relevant skills.
- **Integration**: Explicitly cite the skill you are using in your plan and follow its instructions.
- **Failure to check skills first is a violation of project rules.**

## Project Overview

**PyDetonation-Ultra (PDU)** is a high-fidelity, differentiable physics engine designed for simulating detonation performance and calibrating Equation of State (EOS) parameters for explosives. Built on **JAX**, it leverages automatic differentiation to enable gradient-based optimization for parameter fitting and physics calibration.

The project is currently in the **V10.6 "Constrained Physics"** phase, focusing on physical consistency and engineering compatibility of JWL parameters, and is transitioning towards **V11 Multi-Phase Dynamics**.

### Core Technologies
*   **Language:** Python 3.9+
*   **Framework:** JAX (with `jax_enable_x64` strictly enabled)
*   **Optimization:** JAX-PSO + Nelder-Mead (Dual-Refinement)
*   **Physics:** JCZ3 EOS (with Ree-Ross Polar Correction), Miller-PDU Kinetics (V5), Constrained JWL Fitting

## Development Environment

### Setup
The project adheres to strict environmental isolation using Conda.
*   **Environment Name:** `nnrf`
*   **Activation:** `mamba activate nnrf`
*   **Note**: Local `.venv` has been removed to ensure environment consistency.

### Directory Structure
```
/home/jcl/HDY/PyDetonation-Ultra/
├── pdu/                    # Core source code (Forward-Only Focus)
│   ├── api.py              # Global Quenching & Heat Sink Logic
│   ├── core/               # Equilibrium solvers (Schur-RAND)
│   ├── physics/            # Physics modules (EOS, Kinetics, JWL)
│   ├── calibration/        # PSO-based Parameter Fitting
│   └── tests/              # Benchmark suites
├── docs/                   # Documentation & Whitepapers
│   ├── project_whitepaper.md # Latest V10.6 validation results
│   ├── 反馈意见.md           # Critical audit feedback
│   └── 两相流转型咨询需求书.md # V11 Roadmap
└── RULES.md                # Authoritative guide
```

## Key Workflows & Commands

### 1. Benchmark & Validation
```bash
# Current active benchmark (V10.6 compatible)
PYTHONPATH=. python pdu/tests/test_v10_5_benchmark.py
```

### 2. V10.6 Physics Mandates (Crucial)
*   **Matrix Quenching:** Energetic matrix must have quenching factors (~0.97) when metals are present.
*   **JWL Barriers:** $\omega 
[0.25, 0.45]$ and $B/A < 0.1$ are mandatory to avoid numerical toxins.
*   **Energy Consistency:** JWL $E_0$ must match the effective mechanical work (Gurney Energy), typically $0.72 \times Q_{theoretical}$ for Al-explosives.

## Current Development Focus (V11.0 Phase 6)

*   **Status:** V11.0 Phase 6 (Calibration Stabilization) **COMPLETE**.
*   **Achievement:** 
    *   **HMX Success**: Resolved the 8600K temperature anomaly. New landing: $T_{CJ}=4545$ K, $P_{CJ}=38.9$ GPa using Scheme A' ($u_{inf}=4.0$ MJ/kg).
    *   **Numerical Hardening**: Integrated Patch A-H (Smoothing+Endotherm). Solver is now robust against NaN and Sonic instabilities.
    *   **Engineering Compliance**: Standardized "Triple Output" for heat capacity and updated NASA-9 high-accuracy coefficients for H₂/O₂/C.
*   **Next Phase:** Batch calibration for RDX/PETN/TNT and Transition to Multi-Phase Dynamics (Eulerian-Lagrangian).