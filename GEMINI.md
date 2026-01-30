# PyDetonation-Ultra (PDU) Project Context

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

## Current Development Focus (V11.0 Phase 5)

*   **Status:** V11.0 Phase 5 (Thermodynamic Hardening) Implemented.
*   **Achievement:** Successfully stabilized ZND solver using **Patch A-D** (Smooth Barriers, Sigmoid NASA, Consistent Sound Speed, Reject & Shrink Integrator). Eliminated all NaN/Sonic instabilities.
*   **Current Blocker:** **Reaction Stagnation**. ZND ignition fails due to anomalously low pressure (0.12 GPa vs expected >30 GPa) at VN point. Investigation into unit/state initialization is ongoing.
*   **Next Phase:** Resolve stagnation issue and proceed to Multi-Explosive Benchmark.