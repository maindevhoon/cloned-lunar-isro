# Next Steps — Lunar Autonomy Project (2026)

**Goal**: Evolve the 2024 ISRO BAH winning hackathon prototype into a modern, realistic system using **NVIDIA Isaac Sim** for high-fidelity physics, sensor simulation, and policy training.

This document outlines a phased plan based on analysis of the original codebase.

---

## Current State (as of 2026-06-30)

- Reconstructed and cleaned `main.py` files from the original notebooks.
- Added basic physics simulation in the RL module (velocity, thrust, drag, crater collisions).
- Synthetic data generators so everything runs in Colab without external datasets.
- Good conceptual coverage: ROI (hyperspectral) → Crater detection → Path planning (RL) → Formation.
- Major gaps from 2024:
  - No integration between modules
  - No real physics or sensors
  - Notebook-only code
  - Toy grids instead of realistic terrain

---

## Phase 1: Stabilize & Validate Current Code (1–2 weeks)

### Tasks
- [ ] Upload the project to Colab using the `HOW_TO_COLAB.md` guide and run all demos.
- [ ] Test and fix bugs in the reconstructed `main.py` files (especially DQN/PPO training stability and physics env).
- [ ] Add proper logging, metrics, and visualization improvements.
- [ ] Create simple integration: feed detected craters from `obstacle-detection` into a costmap used by path planning.
- [ ] Write unit tests for the core environments and agents.
- [ ] Improve documentation (docstrings, README examples).

### Deliverables
- Working Colab notebook(s) demonstrating the full pipeline end-to-end.
- A basic `costmap_generator.py` that turns crater detections into a navigation costmap.

### Why this first?
You need a solid baseline before adding complexity with Isaac Sim.

---

## Phase 2: Restructure for Production & Simulation (2–3 weeks)

### Tasks
- [ ] Convert flat folders into a proper Python package:
  ```
  lunar/
  ├── lunar/
  │   ├── perception/
  │   ├── planning/
  │   ├── simulation/
  │   └── data/
  ├── scripts/
  ├── configs/
  └── tests/
  ```
- [ ] Define clean interfaces:
  - `PerceptionOutput` (craters, ROIs, obstacles)
  - `Costmap`
  - `Trajectory`
- [ ] Make the RL environments more modular (separate discrete grid vs physics versions).
- [ ] Add configuration system (YAML or dataclasses) for environments, training, etc.
- [ ] Create synthetic terrain generator that can export heightmaps + crater annotations (reusable for Isaac Sim).

### Deliverables
- Clean, importable package structure.
- `lunar/perception/crater_detector.py` with both CV fallback + model interface.
- Terrain generator that outputs `.npy` heightmaps + crater list.

---

## Phase 3: NVIDIA Isaac Sim Foundation (3–6 weeks)

This is the biggest upgrade.

### 3.1 Environment Setup
- [ ] Install NVIDIA Isaac Sim 4.x (or latest) + Isaac Lab.
- [ ] Create a new repo/folder `lunar_isaac/` or integrate inside this project under `simulation/isaac/`.
- [ ] Set up a basic **Lunar World**:
  - Import real lunar DEM (LRO or Chandrayaan data).
  - Add procedural craters and rocks using the generator from Phase 2.
  - Realistic lighting (Sun angle, long shadows).

### 3.2 Rover + Sensors
- [ ] Add or import a rover asset (simple 4/6-wheel rover).
- [ ] Configure sensors:
  - Stereo RGB cameras
  - LiDAR (for obstacle detection)
  - IMU
  - (Optional) Depth camera

### 3.3 Isaac Lab Environments
- [ ] Create environments that match the old logic:
  - `LunarGridNav-v0` (discrete, for quick iteration)
  - `LunarPhysicsRover-v0` (continuous actions, physics-based, matching the new `RoverPhysicsEnv`)
  - Reward functions ported + improved (progress, safety, energy, slip penalty)
- [ ] Curriculum learning: flat → slopes → crater fields.

### Deliverables
- A working Isaac Sim scene with lunar terrain.
- At least one Isaac Lab task where a rover can navigate to a goal while avoiding craters.
- Video recordings of the rover in simulation.

---

## Phase 4: Close the Loop — Perception + Planning in Sim (4–8 weeks)

### Tasks
- [ ] **Perception in the loop**:
  - Render camera images in Isaac Sim.
  - Run the crater detector (or retrain YOLO-style model) on simulated images.
  - Generate costmaps from detections.
- [ ] **RL Training in Isaac Lab**:
  - Train policies using the costmaps or raw sensor observations.
  - Compare against the old grid-based agents.
- [ ] **ROI Integration**:
  - Simulate material properties or use spectral proxies.
  - Generate "interesting ROI" targets that the planner should visit.
- [ ] **Formation**:
  - Extend to multi-rover in Isaac Sim.

### Integration Architecture (Target)
```
Isaac Sim Sensors → Perception (craters + ROI) → Costmap → 
Global Planner + Local RL Policy → Low-level Control → Rover in Physics
```

---

## Phase 5: Evaluation, Data, and Sim-to-Real (ongoing)

- Define quantitative metrics:
  - Success rate
  - Path efficiency
  - Collision rate
  - Energy / slip
  - Time to reach goal + visit ROIs
- Generate large synthetic datasets in Isaac Sim for perception training (domain randomization).
- Add domain randomization (lighting, texture, sensor noise).
- Plan for real data validation (when possible).

---

## Suggested Priority Order (Right Now)

| Priority | Task | Est. Time | Owner |
|----------|------|-----------|-------|
| 1 | Get the current code running cleanly in Colab | 1–3 days | You |
| 2 | Fix/improve the RL physics environment and add metrics | 3–5 days | You + AI |
| 3 | Basic integration (craters → costmap → planner) | 1 week | You + AI |
| 4 | Restructure into proper package | 1 week | AI assisted |
| 5 | Create Isaac Sim lunar terrain scene + simple rover | 2–3 weeks | Next major milestone |
| 6 | Port physics env to Isaac Lab | 2 weeks | - |

---

## Immediate Next Actions I Can Help With

Tell me which one(s) you want to tackle:

- **A.** Improve the current Python code (bug fixes, better integration, package structure).
- **B.** Generate starter code / configs for NVIDIA Isaac Sim (scene, robot, basic environment).
- **C.** Create a full design document / architecture spec for the Isaac Sim version.
- **D.** Make a Colab-ready notebook that demonstrates the pipeline with nice visualizations.
- **E.** Data pipeline: script to convert real DEMs into Isaac Sim usable terrain.
- **F.** Something else (e.g. add ROS 2 bridge planning, evaluation dashboard, etc.)

---

## Long-term Vision (2026 Target)

A system where:
- You can specify a region using real hyperspectral/DEM data.
- Isaac Sim generates a high-fidelity version of that terrain.
- A rover (or team of rovers) autonomously explores, avoids hazards, and prioritizes ROIs using learned policies trained in simulation.
- Everything is testable, reproducible, and closer to sim-to-real ready.

---

**Status**: This plan is ready to be turned into tasks.

Just reply with the letter or describe what you want to do first (e.g. "Start with B and generate Isaac Sim starter code" or "Let's restructure the package").

We can work step by step from there.