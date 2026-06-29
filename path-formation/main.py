"""
path-formation/main.py
Multi-Rover Path Formation and Coordination

This was empty in the original winning submission.
Here is a clean, runnable implementation that fits the 2024 vision.

Features:
- Generate a base path (A* on costmap or simple spline)
- Create follower paths for a formation (leader + offsets)
- Simple collision avoidance between rovers
- Visualization

Colab:
    !pip install numpy matplotlib scipy
    from path_formation.main import run_formation_demo
    run_formation_demo()
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# =============================================================================
# UTILITIES
# =============================================================================

def generate_base_path(start, goal, width=120, height=60, obstacles=None, steps=25):
    """
    Very simple path generator: straight line + light obstacle repulsion.
    In real system you would use A*, RRT*, or a global planner on a costmap.
    """
    path = [np.array(start, dtype=float)]
    current = np.array(start, dtype=float)
    direction = np.array(goal, dtype=float) - current
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return [start]

    direction /= length
    step_size = length / steps

    for i in range(steps):
        next_pt = current + direction * step_size

        # Simple obstacle avoidance force
        if obstacles is not None:
            for ox, oy, r in obstacles:
                diff = next_pt - np.array([ox, oy])
                dist = np.linalg.norm(diff)
                if dist < r + 6 and dist > 0.1:
                    force = (diff / dist) * (r + 6 - dist) * 0.8
                    next_pt += force

        # bound
        next_pt[0] = np.clip(next_pt[0], 0, width - 1)
        next_pt[1] = np.clip(next_pt[1], 0, height - 1)

        path.append(next_pt)
        current = next_pt

    return path


def smooth_path(path, smoothness=0.8, num_points=60):
    """Spline smoothing for nicer formation paths."""
    pts = np.array(path).T
    if len(pts[0]) < 4:
        return path
    try:
        tck, u = splprep(pts, s=smoothness)
        u_fine = np.linspace(0, 1, num_points)
        x, y = splev(u_fine, tck)
        return list(zip(x, y))
    except Exception:
        return path


# =============================================================================
# FORMATION
# =============================================================================

def create_formation_paths(base_path, num_rovers=4, formation="line", spacing=4.5):
    """
    Given a base path for the leader, create offset paths for followers.
    formation options: "line", "wedge", "echelon"
    """
    base = np.array(base_path)
    paths = [base.copy()]

    if len(base) == 0:
        return [base_path] * num_rovers

    for i in range(1, num_rovers):
        offset_path = []
        for j, p in enumerate(base):
            # Compute tangent
            if j < len(base) - 1:
                tangent = base[j + 1] - p
            else:
                tangent = p - base[j - 1]
            if np.linalg.norm(tangent) < 1e-6:
                tangent = np.array([1.0, 0.0])
            tangent /= (np.linalg.norm(tangent) + 1e-8)

            # perpendicular
            perp = np.array([-tangent[1], tangent[0]])

            if formation == "line":
                offset = perp * spacing * i
            elif formation == "wedge":
                offset = (perp * spacing * (i % 2) * ((i // 2) + 1)) + tangent * spacing * (i // 2) * -0.6
            else:  # echelon
                offset = perp * spacing * i + tangent * spacing * i * 0.3

            new_p = p + offset
            offset_path.append(new_p)
        paths.append(np.array(offset_path))

    return paths


def avoid_collisions(paths, min_dist=3.5):
    """Very basic inter-rover repulsion along the paths."""
    paths = [np.array(p) for p in paths]
    for _ in range(2):  # few relaxation steps
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                for k in range(min(len(paths[i]), len(paths[j]))):
                    diff = paths[i][k] - paths[j][k]
                    d = np.linalg.norm(diff)
                    if d < min_dist and d > 0.01:
                        correction = (diff / d) * (min_dist - d) * 0.5
                        paths[i][k] += correction
                        paths[j][k] -= correction
    return paths


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_formation(base_path, formation_paths, obstacles=None, title="Multi-Rover Formation"):
    plt.figure(figsize=(10, 5))
    colors = ["blue", "red", "green", "orange", "purple"]

    # obstacles
    if obstacles:
        for ox, oy, r in obstacles:
            circle = plt.Circle((ox, oy), r, color="gray", alpha=0.4)
            plt.gca().add_patch(circle)

    # leader
    bp = np.array(base_path)
    plt.plot(bp[:, 0], bp[:, 1], "b-", linewidth=2.5, label="Leader")

    # followers
    for idx, fp in enumerate(formation_paths[1:]):
        fp = np.array(fp)
        plt.plot(fp[:, 0], fp[:, 1], color=colors[(idx + 1) % len(colors)],
                 linewidth=1.8, alpha=0.85, label=f"Rover {idx+2}")

    # start/goal
    plt.scatter([bp[0, 0]], [bp[0, 1]], c="lime", s=90, marker="o", zorder=5, label="Start")
    plt.scatter([bp[-1, 0]], [bp[-1, 1]], c="red", s=110, marker="*", zorder=5, label="Goal")

    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect("equal")
    plt.show()


# =============================================================================
# DEMO
# =============================================================================

def run_formation_demo():
    print("=== Path Formation Demo (Multi-Rover) ===")

    start = (5, 8)
    goal = (105, 42)
    obstacles = [(30, 25, 7), (55, 15, 5), (78, 32, 8), (42, 38, 4)]

    base = generate_base_path(start, goal, width=110, height=55, obstacles=obstacles, steps=30)
    base = smooth_path(base, smoothness=1.2, num_points=55)

    formation_paths = create_formation_paths(base, num_rovers=4, formation="wedge", spacing=5.0)
    formation_paths = avoid_collisions(formation_paths, min_dist=4.0)

    plot_formation(base, formation_paths, obstacles=obstacles)
    print(f"Generated paths for {len(formation_paths)} rovers.")
    print("Formation demo finished.")


if __name__ == "__main__":
    run_formation_demo()
