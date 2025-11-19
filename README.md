# AI-Einstein: Project Master Plan

## 1. Project Overview

**Objective:** Create an AI agent initialized with Newtonian priors (Pre-1905 physics) that, when presented with relativistic trajectory data, "discovers" curved spacetime and derives the Einstein Field Equations.

**The Core Conflict:** The AI will try to fit the data to $F=ma$ (Newton), fail, and then be allowed to learn a Metric Tensor ($g_{\mu\nu}$) to resolve the error.

**The Tech Stack:**
- **Language:** Python 3.10+
- **Deep Learning:** PyTorch (specifically for autograd)
- **Symbolic AI:** PySR (for equation discovery)
- **Math:** NumPy, SciPy (for generating ground truth)

## 2. The Architecture (4 Phases)

### Phase 1: The Universe (Data Generation)
- **Goal:** Create the "Experimental Data."
- **Physics:** Simulate particles passing through a "Dust Cloud" (Interior Schwarzschild Metric).
- **Why:** We need a region where Mass Density ($\rho$) is non-zero to derive $G_{\mu\nu} \propto T_{\mu\nu}$.
- **Output:** `universe_data.csv` containing $(t, x, y, v_x, v_y, \rho)$.

### Phase 2: The Classicist (The Newtonian Failure)
- **Goal:** Prove that Classical Physics cannot explain the data.
- **Model:** A standard Neural Network trained to predict acceleration using Euclidean distance.
- **Result:** We will generate a plot showing the "Newtonian Drift" (Prediction vs Reality error).
- **Milestone:** "The Crisis of Physics" visualization.

### Phase 3: The Geometer (SIREN + Metric Learning)
- **Goal:** Learn the curved metric $g_{\mu\nu}$.
- **Technique:** SIREN (Sinusoidal Representation Network).
- **Logic:**
    1. Learn a coordinate transformation $\Phi: (t, x, y) \to (\tau, X, Y)$.
    2. Force the transformed coordinates to move in straight lines (Newton's 1st Law).
    3. Compute the Metric Tensor $g$ from the Jacobian of $\Phi$.
- **Why SIREN?** Standard ReLU networks have zero 2nd derivatives. Sine waves have infinite smooth derivatives, allowing us to calculate curvature analytically.

### Phase 4: The Theorist (Symbolic Derivation)
- **Goal:** Derive the Equation.
- **Inputs:**
    1. The Einstein Curvature ($G_{00}$) computed via AutoGrad from Phase 3.
    2. The Mass Density ($\rho$) from Phase 1 data.
- **Tool:** PySR.
- **Expected Result:** The AI discovers $G_{00} = k \cdot \rho$.

## 3. Directory Structure

```
AI_Einstein/
├── data/
│   └── universe_data.csv       # Generated in Phase 1
├── models/
│   ├── newton_baseline.pth     # Phase 2 model
│   └── siren_metric.pth        # Phase 3 model
├── src/
│   ├── 1_generator.py          # Simulates the physics
│   ├── 2_newton_fail.py        # Trains the failed Newtonian model
│   ├── 3_siren_training.py     # Trains the Metric Learner
│   ├── 4_curvature_calc.py     # Calculates R_uv from the trained model
│   └── 5_symbolic_discovery.py # Runs PySR
└── README.md
```

## 4. Weekly Schedule

- **Week 1: The Simulation**
    - Write `1_generator.py`.
    - Implement the "Interior Schwarzschild" geodesic equations.
    - Visualize the trajectories passing through the star.
    - **Success Criteria:** A CSV file with curves that look smooth but weird.

- **Week 2: The Newtonian Failure**
    - Write `2_newton_fail.py`.
    - Train a basic MLP on the data.
    - Plot the residuals (Errors).
    - **Success Criteria:** A graph proving "Euclidean geometry is insufficient."

- **Week 3: SIREN Implementation**
    - Write `3_siren_training.py`.
    - Implement the custom SineLayer class in PyTorch.
    - Train the network to "straighten" the trajectories.
    - **Success Criteria:** The Loss converges, and you can visualize the "warped grid."

- **Week 4: Calculus & Derivation**
    - Write `4_curvature_calc.py` using `torch.autograd.functional.jacobian`.
    - Compute the Einstein Tensor values across the grid.
    - Write `5_symbolic_discovery.py` to feed these values into PySR.
    - **Success Criteria:** PySR outputs a linear equation relating Curvature to Density.

## 5. Technical Pitfalls (And how to avoid them)

- **The Singularity:** The math breaks at $r=0$.
    - **Fix:** In `1_generator.py`, remove any particle that gets too close to the center ($r < 0.1$).

- **Training Time:** SIRENs can be slow to converge.
    - **Fix:** Use the Adam optimizer with a learning rate of 1e-4.

- **Symbolic Noise:** PySR might find complex polynomials.
    - **Fix:** Restrict PySR operators to only +, -, *. Do not allow sin, cos, or exp in the final Symbolic Regression step, as the Field Equations are algebraic.
