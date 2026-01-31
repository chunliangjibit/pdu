# PyDetonation-Ultra (PDU) Project Context

## **CRITICAL PROTOCOL: SKILL-FIRST WORKFLOW (MANDATORY)**
**RULE #1:** Whenever you encounter a problem, need to implement a feature, or plan a task, you **MUST FIRST** search the `/home/jcl/HDY/PyDetonation-Ultra/.agent/skills` directory.
- **Action**: Use `find_by_name` or `grep_search` to find relevant skills.
- **Integration**: Explicitly cite the skill you are using in your plan and follow its instructions.
- **Failure to check skills first is a violation of project rules.**

## Project Overview

**PyDetonation-Ultra (PDU)** is a high-fidelity, differentiable physics engine designed for simulating detonation performance and calibrating Equation of State (EOS) parameters for explosives. Built on **JAX**, it leverages automatic differentiation to enable gradient-based optimization for parameter fitting and physics calibration.

The project is currently in the **V11.0 "Multi-Phase Dynamics"** phase, focusing on high-fidelity reactive flow ODEs and unified material calibration.

### Core Technologies
*   **Language:** Python 3.9+
*   **Framework:** JAX (with `jax_enable_x64` strictly enabled)
*   **Acceleration:** GPU-accelerated (RTX 4060 compatible)
*   **Optimization:** JAX-PSO + Nelder-Mead (Dual-Refinement)
*   **Physics:** JCZ3 EOS (with Ree-Ross Polar Correction), Miller-PDU V5, High-T Endotherm (Scheme A')

## Development Environment

### Setup
The project adheres to strict environmental isolation using Conda.
*   **Environment Name:** `nnrf`
*   **Activation:** `mamba activate nnrf`
*   **Note**: Local `.venv` has been removed to ensure environment consistency.

### Directory Structure
```
/home/jcl/HDY/PyDetonation-Ultra/
├── pdu/                    # Core source code
│   ├── api.py              # Global Quenching & ZND Entry
│   ├── core/               # Equilibrium solvers (Schur-RAND)
│   ├── physics/            # Physics modules (EOS, Kinetics, JWL)
│   ├── solver/             # ZND & Stability Framework
│   └── tests/              # Benchmark & Consistency suites
├── documents/               # Reports & Technical Logs
│   ├── dev_log.md          # Project milestones
│   └── 技术反馈落实情况报告_v3.md # Latest engineering validation
└── RULES.md                # Authoritative guide
```

## Key Workflows & Commands

### 1. Thermodynamic Consistency (V11 Mandatory)
```bash
# Verify Cv positivity, derivative consistency, and Ideal Gas limit
PYTHONPATH=. python pdu/tests/test_thermo_consistency.py
```

### 2. HMX Calibration Landing
```bash
# Execute full HMX calibration and ZND profiling
PYTHONPATH=. python pdu/tests/calibrate_hmx_v11.py
```

### 3. Core Physics Mandates (V11 Compliance)
*   **Matrix Quenching:** Matrix efficiency factor (~0.97) for metal-filled systems.
*   **JWL Barriers:** $\omega \in [0.25, 0.45]$ and $B/A < 0.1$ for engineering compatibility.
*   **Triple Output:** Cv diagnostics must include total(J/K), mass(J/kg/K), and molar(J/mol/K).
*   **A-Anchoring:** Energy corrections must be zero-anchored at $T_{ref}=298.15\text{K}$.

## Current Development Focus (V11.0 Phase 6)

*   **Status:** V11.0 Phase 6 (Calibration Stabilization) **COMPLETE**.
*   **Achievement:** 
    *   **HMX Success**: Resolved the 8600K temperature anomaly. New landing: $T_{CJ}=4545$ K, $P_{CJ}=38.9$ GPa using Scheme A' ($u_{inf}=4.0$ MJ/kg).
    *   **Numerical Hardening**: Integrated Patch A-H (Smoothing+Endotherm). Solver is now robust against NaN and Sonic instabilities.
    *   **Engineering Compliance**: Standardized "Triple Output" for heat capacity and updated NASA-9 high-accuracy coefficients for H₂/O₂/C.
*   **Next Phase:** 
    1.  **HMX Closure**: Final landing of all base detonation and JWL EOS parameters.
    2.  **Single Explosive Expansion**: Batch calibration for standard species (RDX, PETN, TNT, etc.) to build a validated material library.
    3.  **Multi-Phase/Composite Dynamics**: Transition to Eulerian-Lagrangian simulations for aluminized and binder-based explosives.