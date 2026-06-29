# Codebase Analysis — Lunar (ISRO BAH 2024 Winning Entry)

**Date**: 2026-06-29  
**Context**: Original winning submission from ISRO Bharat Anveshan Hackathon 2024. Team is revisiting the same track in 2026 with plans to integrate **NVIDIA Isaac Sim** for realistic physics simulation.

---

## Executive Summary

This is a **rapid-prototype hackathon codebase** consisting almost entirely of Google Colab notebooks. It implements a credible end-to-end lunar rover autonomy pipeline using real ISRO data:

- **Site selection** via hyperspectral analysis (Chandrayaan-2 IIRS)
- **Hazard perception** via crater detection
- **Learning-based local navigation** via multiple RL algorithms

**Strengths**:
- Directly uses real mission data
- Covers the full problem (science ROI → perception → planning)
- Shows implementation breadth across CV + RL

**Critical Weaknesses**:
- Zero production code (all `main.py` files are empty)
- No module integration
- No physics, no sensors, no dynamics
- Synthetic navigation environments

**Recommendation**: This is an *excellent conceptual foundation*. The 2026 effort should keep the scientific goals while moving the execution environment to **NVIDIA Isaac Sim + Isaac Lab** as the single source of truth for terrain, perception, and policy training.

---

## Detailed Module Analysis

### 1. roi-identification / hyperspectral.ipynb

**Purpose**: Identify scientifically interesting regions using orbital hyperspectral data.

**Implementation**:
- Loads real Chandrayaan-2 IIRS Level-1 data (`ch2_iir_nci_... .qub` + `.hdr`)
- Geometry file provides per-pixel lat/lon (scan + pixel coordinates)
- Data shape: `(16592, 250, 256)` — a long orbital strip
- Uses `spectral` library (`envi.open`)
- Visualizations:
  - Reflectance curves per (x, y)
  - Multiple band grayscale views
  - ROI cropping with coordinate overlays
- Experiments: PCA, KMeans, RandomForest, simple spectral feature analysis (water absorption bands attempted)

**Data Quality**:
- Real mission data — very strong for a hackathon.
- Geometry alignment is present but basic.

**Gaps**:
- No production pipeline (notebook only)
- No material/mineral classification model saved
- No linkage to ground-level rover view

**Value for 2026**: Keep as source of truth for "where to go". Use Isaac Sim to simulate what a rover would actually see (or proxy spectral signatures via material IDs).

---

### 2. obstacle-detection / crater.ipynb

**Purpose**: Detect craters as primary surface hazards.

**Implementation Details**:
- Dataset: Roboflow-exported crater images (Mars + lunar style) with YOLO-style labels
- `best.pt` weights present (standard YOLO-family output)
- Heavy setup:
  - Cloned `pytorch/vision` at v0.8.2
  - Copied detection references (`engine.py`, `utils.py`, `coco_utils.py`, `transforms.py`)
  - `pycocotools`, `albumentations`, OpenCV
- Visualization code for drawing bounding boxes from labels
- Training/inference scaffolding exists but is fragmented

**Observations**:
- Classic hackathon pattern: get a pre-trained model + dataset + visualize results.
- No clean inference script or exported ONNX/TensorRT artifacts.
- Dataset structure uses `.rf.` filenames (Roboflow).

**Gaps**:
- Not integrated with path planner
- No evaluation on realistic lighting/shadows (critical on the Moon)
- No 3D crater reconstruction from stereo or DEM

**Value for 2026**: Strong starting perception model. Retrain/test inside Isaac Sim with domain randomization (lighting, regolith texture, camera parameters).

---

### 3. path-feedback-rl / rl.ipynb

**Purpose**: Learn safe paths using reinforcement learning.

**Algorithms Implemented** (from scratch):
- Tabular Q-Learning (on downscaled image grid)
- DQN (PyTorch)
- A3C-style worker
- PPO (with clipping + advantage)
- Reward shaping variants (sparse + dense)

**Environment**:
- Custom Gymnasium `GridEnv`
- Binary image: white pixels = obstacles
- Actions: 4-directional
- Start: (0,0), Goal: bottom-right
- Very large synthetic grids in some cells (100k × 30k downscaled)

**Strengths**:
- Good breadth — four different RL approaches compared
- Clean enough to understand the learning dynamics

**Gaps**:
- Completely decoupled from perception output
- No continuous action space or realistic kinematics
- No notion of slope, slip, energy, or time
- Grids are random noise, not real lunar terrain

**Value for 2026**: Excellent reference implementations. Port the best ideas (reward design, curriculum) into Isaac Lab environments with proper rover dynamics.

---

### 4. path-formation /

**Status**: Empty stub (`main.py` is 0 bytes).

**Likely Intention**:
- Multi-rover formation flying / driving
- Coordinated path generation
- Communication-aware planning

**Recommendation**: Re-scope for 2026 as "multi-agent coordination in Isaac Sim" once single-agent navigation is solid.

---

## Cross-Cutting Issues

| Area              | Current State                     | Impact                              | Priority for 2026 |
|-------------------|-----------------------------------|-------------------------------------|-------------------|
| Code Organization | Notebooks only                    | Not maintainable, not testable      | High             |
| Integration       | None between modules              | End-to-end story is broken          | High             |
| Terrain Fidelity  | Random binary grids               | Policies won't transfer             | Critical         |
| Physics           | None                              | No wheel slip, dynamics, power      | Critical         |
| Sensors           | None                              | No realistic perception training    | High             |
| Evaluation        | Visual inspection + print(path)   | No quantitative metrics             | Medium           |
| Reproducibility   | Hardcoded Drive paths             | Fragile                             | High             |

---

## Technical Stack (2024)

- Python + Jupyter (Colab)
- `gymnasium`, `torch`, `numpy`, `matplotlib`, `PIL`, `scikit-learn`, `spectral`
- Custom RL from scratch + torchvision detection references
- Real ISRO data + public crater datasets

---

## Recommended 2026 Architecture with NVIDIA Isaac Sim

### Guiding Principle
**Isaac Sim becomes the central truth** for terrain, sensors, physics, and training. Notebooks become experiments and analysis tools, not the core system.

### Layered Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Data & Ground Truth                     │
│  • Chandrayaan-2 hyperspectral + geometry                  │
│  • Lunar DEMs (LRO, Chandrayaan)                           │
│  • Crater catalogs                                         │
└───────────────────────────────┬────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────┐
│              Isaac Sim World Generation                    │
│  • Import DEM + overlay craters                            │
│  • Procedural regolith, rocks, slopes                      │
│  • Lighting model (Sun angle, long shadows)                │
└───────────────────────────────┬────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌──────────────────┐    ┌────────────────┐
│  Perception   │     │   Sensors        │    │   Materials    │
│  Module       │ ←── │   (RGB, Depth,   │ ←── │   (for spectral│
│  (Crater Det, │     │    LiDAR, etc.)  │    │    proxy)      │
│   ROI proxy)  │     └──────────────────┘    └────────────────┘
└───────┬───────┘
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│                 Planning & Control Stack                   │
│  • Global planner (A*, RRT, etc on costmap)                │
│  • Local RL policy (trained in Isaac Lab)                  │
│  • Formation controller (later)                            │
└───────┬────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│                   Isaac Lab RL Environments                │
│  • Rover model with realistic wheels + suspension          │
│  • Curriculum: flat → slopes → crater fields               │
│  • Reward: progress + safety + energy + slip penalty       │
└────────────────────────────────────────────────────────────┘
```

### Migration Priorities

1. **Foundation**
   - Set up Isaac Sim project + lunar asset pipeline
   - Reproduce a basic cratered terrain from public DEM

2. **Perception in the Loop**
   - Render camera images in Isaac Sim
   - Run existing crater model (or fine-tune)
   - Generate labeled datasets with domain randomization

3. **RL in Isaac Lab**
   - Create a rover locomotion + navigation task
   - Port reward ideas from the 2024 PPO notebook
   - Train policies that respect physics (no instant 90° turns)

4. **Integration Layer**
   - Perception → costmap / obstacle map
   - Costmap → global planner
   - Planner + local policy closed loop

5. **Advanced**
   - Hyperspectral material classification proxy
   - Multi-rover formation
   - Sim-to-real validation plan

---

## Suggested Repository Structure (Target)

```
lunar-2026/
├── README.md
├── ANALYSIS.md
├── docs/
│   └── architecture.md
├── data/
│   └── (gitignored or LFS)
├── perception/
│   ├── crater_detector/
│   └── spectral_roi/
├── simulation/
│   ├── isaac/
│   │   ├── terrain/
│   │   ├── sensors/
│   │   └── environments/
│   └── isaac_lab/
│       └── tasks/
├── planning/
│   ├── global/
│   └── local_rl/
├── control/
├── training/
│   └── scripts/
├── evaluation/
│   └── metrics/
└── scripts/
```

---

## Immediate Next Steps (Recommended)

1. **Today/This week**
   - Replace current README (done)
   - Create this ANALYSIS.md (in progress)
   - Decide on Isaac Sim version + hardware (local vs cloud)

2. **Next 2 weeks**
   - Stand up a minimal Isaac Sim lunar scene with one crater field
   - Export a few camera renders
   - Run the 2024 crater model on those renders (sanity check)

3. **Next sprint**
   - Implement first Isaac Lab task (flat terrain goal reaching)
   - Define success metrics (path efficiency, collision rate, energy)

---

## Conclusion

The 2024 codebase is a **strong conceptual win** that correctly identified the three pillars of the problem:

- Where should the rover go? (ROI / hyperspectral)
- What is dangerous? (Crater detection)
- How do we get there safely? (RL navigation)

What it lacked was **fidelity and integration**.

NVIDIA Isaac Sim directly solves the fidelity problem and provides a natural place to integrate everything. The 2026 project has a clear and exciting path: take the scientific intent of the winning hackathon entry and realize it with modern robotics simulation.

---

**Maintained by**: 2026 team  
**Original credit**: 2024 ISRO BAH winning team
