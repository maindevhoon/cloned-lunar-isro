# Lunar — ISRO BAH 2024 Winning Hackathon Project

Autonomous navigation and site selection pipeline for lunar rover missions.

**Track**: ISRO Bharat Anveshan Hackathon (BAH) 2024 — Lunar exploration / rover autonomy  
**Status**: Winning prototype (2024). Notebook-heavy research codebase.  
**Goal (2026 refresh)**: Evolve into a production-grade system using **NVIDIA Isaac Sim** for high-fidelity physics and sensor simulation.

---

## Overview

This project demonstrates a modular pipeline for lunar surface operations:

1. **ROI Identification** — Analyze real Chandrayaan-2 IIRS hyperspectral data to find scientifically valuable regions.
2. **Hazard Detection** — Deep learning-based crater detection on surface imagery.
3. **Safe Path Planning** — Reinforcement learning agents that learn to navigate obstacle-rich terrain.
4. **Path Formation** — (Placeholder) Intended for multi-rover coordination and trajectory generation.

The original submission was built rapidly in Google Colab using real ISRO data + modern ML techniques.

---

## Repository Structure

```
lunar/
├── README.md
├── ANALYSIS.md                     # Detailed technical analysis + Isaac Sim roadmap
├── obstacle-detection/
│   ├── crater.ipynb                # Crater / hazard detection (YOLO-style + Roboflow dataset)
│   └── main.py                     # (empty — logic in notebook)
├── roi-identification/
│   ├── hyperspectral.ipynb         # Chandrayaan-2 IIRS hyperspectral ROI analysis
│   └── main.py                     # (empty)
├── path-feedback-rl/
│   ├── rl.ipynb                    # Q-learning, DQN, A3C, PPO for grid navigation
│   └── main.py                     # (empty)
├── path-formation/
│   └── main.py                     # Placeholder (multi-agent / formation control)
└── ...
```

---

## Core Modules

| Module                  | Focus                              | Key Technologies                          | Data Sources                     |
|-------------------------|------------------------------------|-------------------------------------------|----------------------------------|
| `roi-identification`    | Scientific site selection          | `spectral`, PCA, KMeans, RandomForest     | Chandrayaan-2 IIRS (real)        |
| `obstacle-detection`    | Crater & hazard perception         | PyTorch, torchvision detection refs, best.pt (YOLO-style) | Roboflow Mars/Lunar crater dataset |
| `path-feedback-rl`      | Learning-based navigation          | Gymnasium, PyTorch (DQN / A3C / PPO)      | Synthetic obstacle grids         |
| `path-formation`        | Multi-rover coordination           | — (stub)                                  | —                                |

### Highlights from Original Work

- Real Chandrayaan-2 hyperspectral strip processing (16k+ scan lines, 256 bands).
- Crater detection pipeline with pre-trained weights + visualization.
- Multiple RL algorithms compared on the same grid navigation task.
- Strong coverage of the end-to-end problem (science → perception → planning).

---

## Limitations of the 2024 Prototype

- All logic lives in Colab notebooks; `main.py` files are empty.
- No integration between modules (craters are not fed into the RL planner).
- Navigation uses toy random grids, not real elevation or detected hazards.
- Zero physics simulation, sensor noise, wheel-terrain dynamics, lighting, or power modeling.
- Difficult to scale training or evaluate closed-loop behavior.

---

## 2026 Direction: NVIDIA Isaac Sim Integration

We are rebuilding on the same track with a major upgrade: **bringing realistic physics and simulation into the loop using NVIDIA Isaac Sim + Isaac Lab**.

### Why Isaac Sim?

- High-fidelity lunar regolith + crater physics (PhysX).
- Accurate rover dynamics (slip, suspension, low-g behavior).
- Rich sensor suite (cameras, LiDAR, IMU) with realistic noise/lighting.
- Domain randomization for robust perception training.
- Native support for large-scale RL (Isaac Lab).
- Sim-to-real pathway.

### Proposed Architecture (Target)

```
Real ISRO Data + DEMs
        │
        ▼
┌───────────────────────┐
│   Terrain Generation  │  ← Isaac Sim (craters, rocks, regolith)
└───────────┬───────────┘
            │
    ┌───────┴────────┐
    ▼                ▼
┌──────────────┐  ┌─────────────────────┐
│  Perception  │  │  Isaac Sim Sensors  │
│ (Crater CV + │← │  (stereo, LiDAR)    │
│ Hyperspectral│  └─────────────────────┘
│  models)     │
└───────┬──────┘
        │
        ▼
┌──────────────────────────────┐
│   Global + Local Planning    │
│   (Hybrid classical + RL)    │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│  Isaac Lab RL Training               │
│  (PPO / SAC + custom lunar rewards)  │
└──────────────────────────────────────┘
                │
                ▼
         Multi-Rover Formation
```

### Planned Work

- [ ] Import real lunar DEMs + generate procedural craters in Isaac Sim
- [ ] Port / retrain crater detector inside the simulator
- [ ] Create Gymnasium/Isaac wrappers for the old RL environments
- [ ] Train physics-aware navigation policies
- [ ] Add hyperspectral proxy via material classification + sensor simulation
- [ ] Multi-agent formation control
- [ ] Evaluation harness (success rate, energy, safety, slip)
- [ ] Sim-to-real transfer experiments

---

## Getting Started (2026 Reconstructed Version)

The `main.py` files have been filled with clean, runnable implementations based on the original notebooks.

### Quick run in Google Colab

```python
# Paste this in a cell
!pip install -q gymnasium torch numpy matplotlib opencv-python-headless scikit-learn scipy

# Upload the .py files (or %run them if you have the folder structure)
import colab_runner
colab_runner.run_everything()
```

You can also execute individual modules:
- `path-feedback-rl/main.py` → includes discrete grid + **simple Newtonian physics** rover
- `roi-identification/main.py`
- `obstacle-detection/main.py`
- `path-formation/main.py`

**Note on "realistic physics"**: The original LinkedIn claims were aspirational. The 2024 code was grid-based RL with no real physics. We added a basic continuous physics layer (velocity, thrust, drag, crater collisions) so you can demonstrate "physics" immediately. True high-fidelity simulation (regolith, wheel slip, lighting, sensors) is best done in **NVIDIA Isaac Sim**.

## Original Notebooks

The notebooks still contain the raw experiments and can be useful for reference.

---

## Roadmap (2026)

See `ANALYSIS.md` for the full technical deep-dive and detailed migration plan.

High-level goals:
- Move core logic out of notebooks into clean, testable Python modules.
- Make Isaac Sim the primary development, training, and evaluation environment.
- Achieve closed-loop autonomy: perception → planning → control in simulation.
- Target reproducibility and hardware-in-the-loop readiness.

---

## Acknowledgments

- ISRO for the hackathon and Chandrayaan-2 data access.
- Original 2024 team for the winning prototype.
- NVIDIA Isaac Sim / Isaac Lab team for the simulation platform.

---

**License**: TBD (research / hackathon origins)

Contributions welcome as we evolve this toward a realistic Isaac Sim-based lunar autonomy stack.
