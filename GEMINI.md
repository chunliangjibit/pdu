# PyDetonation-Ultra (PDU) Project Context

## Project Overview

**PyDetonation-Ultra (PDU)** is a high-fidelity, differentiable physics engine designed for simulating detonation performance and calibrating Equation of State (EOS) parameters for explosives. Built on **JAX**, it leverages automatic differentiation to enable gradient-based optimization for inverse design (recipe formulation) and parameter fitting.

The project is currently in the **V10.5 "Thermal Lag & Relaxed Anchor"** phase, focusing on resolving long-standing discrepancies in modeling non-ideal aluminized explosives (like Tritonal and PBXN-109) and stabilizing JWL parameter fitting.

### Core Technologies
*   **Language:** Python 3.9+
*   **Framework:** JAX (with `jax_enable_x64` strictly enabled for thermodynamic precision)
*   **Optimization:** Optax, SciPy, custom Particle Swarm Optimization (PSO)
*   **Physics:** ZND Model, JWL EOS, Miller-PDU Kinetic Models, Murnaghan Solid EOS

## Development Environment

### Setup
The project adheres to strict environmental isolation using Conda.

*   **Environment Name:** `nnrf`
*   **Activation:** `mamba activate nnrf`
*   **Key Dependencies:** `jax`, `jaxlib`, `numpy`, `scipy`, `optax`, `matplotlib`, `pandas`.

### Directory Structure
```
/home/jcl/HDY/PyDetonation-Ultra/
├── pdu/                    # Core source code
│   ├── api.py              # Main entry point (Afterburning Injection & Q-Anchor logic)
│   ├── core/               # Equilibrium solvers (implicit differentiation)
│   ├── physics/            # Physics modules (EOS, Kinetics, JWL)
│   │   ├── kinetics.py     # Thermal Lag & Miller models
│   │   └── jwl.py          # JWL fitting (6-Param Relaxed PSO + Energy Soft Constraint)
│   └── tests/              # Benchmark suites
├── docs/                   # Documentation & Whitepapers
│   ├── rules.md            # CRITICAL: Development rules & physics mandates
│   ├── 专家意见.md          # Expert feedback driving V10.5 changes
│   └── *_benchmark.py      # Benchmark scripts
├── pyproject.toml          # Project configuration
└── RULES.md                # The authoritative guide for this project
```

## Key Workflows & Commands

### 1. Benchmark & Validation
The primary method for verifying physics changes is running the version-specific benchmark script.
```bash
# Current active benchmark (V10.5)
PYTHONPATH=. python pdu/tests/test_v10_5_benchmark.py
```

### 2. Physics Mandates (Crucial)
*   **Precision:** `jax.config.update("jax_enable_x64", True)` is mandatory.
*   **Thermal Lag:** For aluminized explosives, the "Thermal Lag" model (`kinetics.py`) must be used, freezing aluminum reaction at the CJ plane.
*   **Afterburning Injection:** Energy from unreacted Al must be injected into the expansion tail ($V > 1.5$) in `api.py` to allow JWL to fit high-work trails.
*   **Energy Anchor Correctness:** JWL energy target ($E_0$) must be based on **Total Heat of Detonation ($Q \times \rho_0$)** and anchored at the **CJ state ($V_{CJ}$)**, not at $V=1$ or using $U_{form}$.
*   **6-Param PSO:** Always decouple parameter $C$ from hard energy formulas during optimization to avoid initialization death zones.

## Current Development Focus (V10.5)

*   **Objective:** Implement "Expert Feedback" regarding non-ideal detonation and JWL robustness.
*   **Key Change 1 (Kinetics):** Implemented Thermal Lag ($	au_{ind} \sim \mu s$) to enforce inert Al behavior at the nanosecond CJ plane.
*   **Key Change 2 (Afterburning):** Added sigmoid-based energy injection for $V \in [1.5, 7.0]$ to simulate late-time work.
*   **Key Change 3 (JWL Fit):** Overhauled `jwl.py` to use a 6-parameter PSO search with energy and sound-speed soft constraints, successfully preventing parameter collapse and allowing healthy $B$ values.