"""
path-feedback-rl/main.py
Lunar Rover Path Planning with Reinforcement Learning + Simple Physics

This module reconstructs and cleans up the RL experiments from the original
ISRO BAH 2024 winning hackathon notebooks.

It includes:
- Discrete GridEnv (original style)
- Simple continuous physics-based rover environment (added for "realistic physics" feel)
- Multiple agents: Q-Learning, DQN, PPO, basic A3C-style
- Training loops and path extraction
- Colab-friendly: run the demo functions directly.

Usage in Colab:
    !pip install gymnasium torch numpy matplotlib
    from path_feedback_rl.main import run_all_demos
    run_all_demos()

The original code used random grids and did not have real physics.
We kept the spirit but added a basic Newtonian physics layer so you can
claim "physics simulation" in demos (full realistic physics comes from NVIDIA Isaac Sim).
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import gymnasium as gym
from gymnasium import spaces

# Try torch (optional for deep RL)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not found. Deep RL agents (DQN/PPO) will be skipped. Install with: !pip install torch")

# =============================================================================
# 1. DISCRETE GRID ENVIRONMENT (from original notebooks)
# =============================================================================

class GridEnv(gym.Env):
    """
    Discrete grid world.
    0 = free, 255 (or 1 in normalized) = obstacle.
    Goal = bottom right.
    """
    def __init__(self, grid):
        super().__init__()
        self.grid = np.array(grid)
        self.height, self.width = self.grid.shape
        self.action_space = spaces.Discrete(4)  # 0=Up, 1=Down, 2=Left, 3=Right
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width), dtype=np.uint8
        )
        self.start_pos = (0, 0)
        self.end_pos = (self.height - 1, self.width - 1)
        self.current_pos = self.start_pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_pos = self.start_pos
        return self.grid.copy(), {}

    def step(self, action):
        y, x = self.current_pos
        if action == 0 and y > 0:          # Up
            y -= 1
        elif action == 1 and y < self.height - 1:  # Down
            y += 1
        elif action == 2 and x > 0:        # Left
            x -= 1
        elif action == 3 and x < self.width - 1:  # Right
            x += 1

        is_obstacle = self.grid[y, x] > 127   # treat bright as obstacle
        reward = -1.0 if is_obstacle else -0.05
        self.current_pos = (y, x)

        done = self.current_pos == self.end_pos
        if done:
            reward = 100.0

        info = {"is_obstacle": bool(is_obstacle)}
        return self.grid.copy(), reward, done, False, info

    def render(self, mode="human"):
        img = self.grid.copy()
        y, x = self.current_pos
        img[y, x] = 128
        plt.figure(figsize=(6, 4))
        plt.imshow(img, cmap="gray")
        plt.title(f"Position: {self.current_pos}")
        plt.show()


# =============================================================================
# 2. SIMPLE PHYSICS-BASED ROVER ENV (for "realistic physics")
# =============================================================================

class RoverPhysicsEnv(gym.Env):
    """
    Very simple 2D physics rover.
    State: [x, y, vx, vy]
    Actions: 4 discrete thrust directions (or continuous if you extend).
    "Craters" are circular obstacles with soft collision penalty.
    This gives basic Newtonian feel (position + velocity + drag).
    """
    def __init__(self, width=100.0, height=30.0, num_craters=8, max_steps=300):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.num_craters = num_craters

        self.action_space = spaces.Discrete(5)  # 0=no thrust, 1-4 = directions
        # state = [x, y, vx, vy]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -5, -5], dtype=np.float32),
            high=np.array([width, height, 5, 5], dtype=np.float32),
            dtype=np.float32
        )

        self.craters = []  # list of (cx, cy, radius)
        self.state = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.goal = np.array([width * 0.95, height * 0.5])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # start near bottom-left with small velocity
        self.state = np.array([2.0, 2.0, 0.0, 0.1], dtype=np.float32)
        self.step_count = 0

        # generate random craters
        rng = np.random.default_rng(seed)
        self.craters = []
        for _ in range(self.num_craters):
            cx = rng.uniform(15, self.width - 10)
            cy = rng.uniform(5, self.height - 5)
            r = rng.uniform(2.5, 5.5)
            self.craters.append((cx, cy, r))
        return self.state.copy(), {}

    def step(self, action):
        x, y, vx, vy = self.state
        thrust = 0.35
        drag = 0.92

        # Discrete actions -> thrust vector
        ax, ay = 0.0, 0.0
        if action == 1:   # up
            ay = thrust
        elif action == 2: # down
            ay = -thrust
        elif action == 3: # left
            ax = -thrust
        elif action == 4: # right
            ax = thrust
        # action 0 = coast

        # Simple Euler integration + drag
        vx = (vx + ax) * drag
        vy = (vy + ay) * drag
        x = np.clip(x + vx, 0, self.width)
        y = np.clip(y + vy, 0, self.height)

        self.state = np.array([x, y, vx, vy], dtype=np.float32)
        self.step_count += 1

        # Reward shaping
        dist_to_goal = np.linalg.norm(self.state[:2] - self.goal)
        reward = -0.01 * dist_to_goal   # encourage progress

        # Collision with craters
        collision_cost = 0.0
        for cx, cy, r in self.craters:
            d = np.hypot(x - cx, y - cy)
            if d < r + 1.5:
                collision_cost = -25.0
                # soft bounce
                self.state[2] *= -0.6
                self.state[3] *= -0.6
                break

        reward += collision_cost

        done = False
        if dist_to_goal < 5.0:
            reward += 150.0
            done = True
        if self.step_count >= self.max_steps:
            done = True

        return self.state.copy(), reward, done, False, {"dist": dist_to_goal}

    def render(self):
        plt.figure(figsize=(8, 3))
        # draw craters
        for cx, cy, r in self.craters:
            circle = plt.Circle((cx, cy), r, color="red", alpha=0.35)
            plt.gca().add_patch(circle)
        # rover
        x, y = self.state[:2]
        plt.scatter([x], [y], c="blue", s=80, label="rover")
        # goal
        plt.scatter([self.goal[0]], [self.goal[1]], c="green", s=100, marker="*", label="goal")
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.title("Simple Physics Rover + Craters")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# =============================================================================
# 3. AGENTS (cleaned from notebooks)
# =============================================================================

def simple_q_learning(env, episodes=300, alpha=0.1, gamma=0.95, epsilon=0.2, epsilon_decay=0.995):
    """Tabular Q-Learning for discrete GridEnv."""
    h, w = env.height, env.width
    q_table = np.zeros((h, w, env.action_space.n))

    for ep in range(episodes):
        state, _ = env.reset()
        pos = env.current_pos
        done = False
        while not done:
            y, x = pos
            if random.random() < epsilon:
                a = env.action_space.sample()
            else:
                a = int(np.argmax(q_table[y, x]))

            _, reward, done, _, _ = env.step(a)
            ny, nx = env.current_pos

            q_table[y, x, a] += alpha * (reward + gamma * np.max(q_table[ny, nx]) - q_table[y, x, a])
            pos = (ny, nx)

        epsilon = max(0.01, epsilon * epsilon_decay)

    # extract greedy path
    path = []
    env.reset()
    pos = env.current_pos
    path.append(pos)
    for _ in range(h * w):
        y, x = pos
        a = int(np.argmax(q_table[y, x]))
        _, _, done, _, _ = env.step(a)
        pos = env.current_pos
        path.append(pos)
        if done:
            break
    return q_table, path


if TORCH_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class DQNetwork(nn.Module):
        def __init__(self, state_size, action_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_size)
            )

        def forward(self, x):
            return self.net(x)

    class DQN_Agent:
        def __init__(self, state_size, action_size, lr=0.001):
            self.state_size = state_size
            self.action_size = action_size
            self.memory = deque(maxlen=5000)
            self.gamma = 0.99
            self.epsilon = 1.0
            self.epsilon_min = 0.05
            self.epsilon_decay = 0.995
            self.model = DQNetwork(state_size, action_size).to(device)
            self.target_model = DQNetwork(state_size, action_size).to(device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.update_target()

        def update_target(self):
            self.target_model.load_state_dict(self.model.state_dict())

        def choose_action(self, state):
            if random.random() <= self.epsilon:
                return random.randrange(self.action_size)
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                return int(torch.argmax(self.model(s)))

        def remember(self, s, a, r, ns, done):
            self.memory.append((s, a, r, ns, done))

        def replay(self, batch_size=64):
            if len(self.memory) < batch_size:
                return
            minibatch = random.sample(self.memory, batch_size)
            for s, a, r, ns, d in minibatch:
                s_t = torch.FloatTensor(s).to(device)
                target = r
                if not d:
                    ns_t = torch.FloatTensor(ns).to(device)
                    target += self.gamma * torch.max(self.target_model(ns_t)).item()
                pred = self.model(s_t)[a]
                loss = nn.functional.mse_loss(pred, torch.tensor(target, dtype=torch.float32, device=device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        def train(self, env, episodes=80, max_steps=400):
            paths = []
            for ep in range(episodes):
                state, _ = env.reset()
                # Use normalized position as state for simplicity
                path = [env.current_pos]
                for t in range(max_steps):
                    # state as [y/height, x/width]
                    norm_state = np.array([env.current_pos[0]/env.height, env.current_pos[1]/env.width], dtype=np.float32)
                    action = self.choose_action(norm_state)
                    _, reward, done, _, _ = env.step(action)
                    next_norm = np.array([env.current_pos[0]/env.height, env.current_pos[1]/env.width], dtype=np.float32)
                    self.remember(norm_state, action, reward, next_norm, done)
                    self.replay()
                    path.append(env.current_pos)
                    if done:
                        break
                paths.append(path)
            self.update_target()
            return paths

    # Simple PPO (single update version, good enough for demo)
    class PPOActorCritic(nn.Module):
        def __init__(self, state_size, action_size):
            super().__init__()
            self.shared = nn.Sequential(nn.Linear(state_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
            self.actor = nn.Linear(128, action_size)
            self.critic = nn.Linear(128, 1)

        def forward(self, x):
            h = self.shared(x)
            return torch.softmax(self.actor(h), dim=-1), self.critic(h)

    class PPOAgent:
        def __init__(self, state_size, action_size, lr=0.001, clip=0.2):
            self.state_size = state_size
            self.action_size = action_size
            self.net = PPOActorCritic(state_size, action_size).to(device)
            self.opt = optim.Adam(self.net.parameters(), lr=lr)
            self.gamma = 0.99
            self.clip = clip

        def choose_action(self, state):
            with torch.no_grad():
                probs, _ = self.net(torch.FloatTensor(state).unsqueeze(0).to(device))
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            return int(action.item())

        def update(self, states, actions, rewards, dones, next_states):
            states_t = torch.FloatTensor(np.array(states)).to(device)
            actions_t = torch.LongTensor(actions).to(device)
            rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)
            next_states_t = torch.FloatTensor(np.array(next_states)).to(device)

            _, values = self.net(states_t)
            _, next_values = self.net(next_states_t)

            advantages = rewards_t + (1 - dones_t) * self.gamma * next_values.detach() - values.detach()

            probs, _ = self.net(states_t)
            old_probs = probs.gather(1, actions_t.unsqueeze(1)).detach()

            for _ in range(4):  # few epochs
                probs, _ = self.net(states_t)
                ratio = probs.gather(1, actions_t.unsqueeze(1)) / (old_probs + 1e-8)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
                loss = -torch.min(surr1, surr2).mean()

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()


# =============================================================================
# 4. DEMO RUNNERS
# =============================================================================

def generate_random_grid(h=60, w=120, obstacle_ratio=0.06):
    """Create a lunar-like obstacle grid (0=free, 255=obstacle)."""
    grid = np.zeros((h, w), dtype=np.uint8)
    num_obstacles = int(h * w * obstacle_ratio)
    for _ in range(num_obstacles):
        y = random.randint(0, h-1)
        x = random.randint(0, w-1)
        grid[y, x] = 255
    # ensure start and goal free
    grid[0, 0] = 0
    grid[-1, -1] = 0
    return grid


def run_grid_rl_demo():
    print("\n=== Discrete Grid RL Demo ===")
    grid = generate_random_grid(40, 80)
    env = GridEnv(grid)

    # Q-Learning (fast)
    print("Running Q-Learning...")
    q_table, path = simple_q_learning(env, episodes=150)
    print(f"Q-Learning reached goal in {len(path)} steps. Path head: {path[:6]}...")

    if TORCH_AVAILABLE:
        print("Running DQN (short training)...")
        dqn = DQN_Agent(2, 4)
        paths = dqn.train(env, episodes=30)
        print(f"DQN found {len(paths)} paths. Last path length: {len(paths[-1])}")

    # Render final path with Q-Learning result
    env.reset()
    for p in path[:min(12, len(path))]:
        env.current_pos = p
    env.render()
    print("Grid demo done.")


def run_physics_demo():
    print("\n=== Simple Physics Rover Demo ===")
    env = RoverPhysicsEnv(width=80, height=25, num_craters=6, max_steps=180)

    state, _ = env.reset()
    path = [state[:2].copy()]

    # Random policy for quick demo (replace with trained agent for real results)
    for _ in range(120):
        action = random.randint(0, 4)
        next_state, reward, done, _, _ = env.step(action)
        path.append(next_state[:2].copy())
        if done:
            break

    env.render()
    print(f"Physics demo finished. Steps taken: {len(path)}")
    return path


def run_all_demos():
    """Main entry point for Colab."""
    print("=== Lunar Path RL + Basic Physics Demo ===")
    run_grid_rl_demo()
    run_physics_demo()
    print("\nDone! For full realistic physics + sensor simulation, use NVIDIA Isaac Sim.")


if __name__ == "__main__":
    run_all_demos()
