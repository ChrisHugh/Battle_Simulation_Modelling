import math
import random
from typing import Optional, Tuple, Dict, Any

import numpy as np

# Gymnasium API (Python 3.11 compatible)
import gymnasium as gym
from gymnasium import spaces

# Local simulation entities (pygame-backed drawing kept intact)
from entities import Boss, Agent, WIDTH, HEIGHT


class BattleArenaEnv(gym.Env):
    """Gym 0.21-compatible environment wrapping the boss-vs-agent simulator.

    Notes for learning:
    - Actions are discrete with 9 choices (no-op + 8 move directions). Attacks are handled by the Agent logic.
    - Observations are a normalized vector (agent-centric and boss-centric features).
    - Reward is dense: +damage dealt, -damage taken, +small survival, terminal bonuses.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self,
                 max_steps: int = 1000,
                 role: str = "warrior",
                 headless: bool = True,
                 seed: Optional[int] = None):
        super().__init__()
        self.max_steps = max_steps
        self.role = role
        self.headless = headless

        # Discrete 10: no-op, 8 directions, attack
        self.action_space = spaces.Discrete(10)

        # Observation vector (12 dims):
        # agent_hp, agent_x, agent_y,
        # boss_hp, rel_x, rel_y, dist,
        # active_ability_id (0-3 normalized), ability_ticks_remaining (0..1),
        # step_progress (0..1), speed_norm (0..1)
        # NOTE: compact to keep learnable; you can extend later.
        low = np.array([0.0, 0.0, 0.0,
                        0.0, -1.0, -1.0, 0.0,
                        0.0, 0.0,
                        0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0,
                         1.0, 1.0, 1.0, 1.0,
                         1.0, 1.0,
                         1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Internal state
        self.boss: Optional[Boss] = None
        self.agent: Optional[Agent] = None
        self.num_steps = 0
        self.prev_boss_health = None
        self.prev_agent_health = None
        self.prev_distance_norm = None  # For distance-based shaping

        # Rendering (lazy init to avoid pygame overhead in training)
        self._pygame_init_done = False
        self._screen = None
        self._clock = None
        self._font = None
        self.overlay_text: Optional[str] = None  # Set by trainer for on-screen info
        self.action_history = []
        self.max_history = 20

        if seed is not None:
            self.seed(seed)

    def seed(self, seed: Optional[int] = None):
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    # --- Gymnasium API ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)
        # Deterministic spawn positions
        self.boss = Boss(x=WIDTH // 2, y=100, health=1000)
        self.agent = Agent(x=random.randint(300,500) // 2, y=random.randint(200,400), role=self.role)

        self.num_steps = 0
        self.prev_boss_health = self.boss.health
        self.prev_agent_health = self.agent.health
        # Initialize previous distance (normalized) for shaping
        initial_dist = math.hypot(self.boss.x - self.agent.x, self.boss.y - self.agent.y)
        self.prev_distance_norm = min(1.0, initial_dist / max(WIDTH, HEIGHT))

        obs = self._get_observation()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: int):
        assert self.boss is not None and self.agent is not None, "Call reset() before step()"

        # Track history of actions
        self.action_history.append(action)
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)

        if action == 9:
            self.agent.attack_or_heal(self.boss, [self.agent])
        else:
        # Map discrete action to direction unit vector
            dx, dy = self._action_to_unit_vector(action)
            # Move agent using a goal point one step in the chosen direction
            goal_x = self.agent.x + dx
            goal_y = self.agent.y + dy
            self.agent.move(goal_x=goal_x, goal_y=goal_y)

        # Boss AI step
        self.boss.step([self.agent])

        # Compute reward components
        damage_dealt = max(0, (self.prev_boss_health - self.boss.health))
        damage_taken = max(0, (self.prev_agent_health - self.agent.health))

        # Scale and clip
        reward = 0.0
        reward += min(3, 0.1 * damage_dealt)
        reward -= min(10, 0.5 * damage_taken)
        reward += 0.001  # survival shaping

        # Distance shaping (potential-based): reward progress towards boss
        current_dist = math.hypot(self.boss.x - self.agent.x, self.boss.y - self.agent.y)
        current_dist_norm = min(1.0, current_dist / max(WIDTH, HEIGHT))


        # Role-specific optimal distance rewards
        if self.agent.role == 'archer':
            optimal_min, optimal_max = 120, 240  # Archer's sweet spot
            if optimal_min <= current_dist <= optimal_max:
                # Reward staying in optimal range
                reward += 0.05  # Small bonus for being in position
                # Reward moving INTO optimal range
                if self.prev_distance_norm is not None:
                    dist_change = (self.prev_distance_norm - current_dist_norm) * max(WIDTH, HEIGHT)
                    if dist_change > 0:  # Moving closer
                        reward += 0.02
            else:
                # Penalty for being outside optimal range
                reward -= 0.01

        elif self.agent.role == 'warrior':
            optimal_max = 40  # Warriors want to be close
            if current_dist <= optimal_max:
                reward += 0.05  # Bonus for being in melee range
                if self.prev_distance_norm is not None:
                    dist_change = (self.prev_distance_norm - current_dist_norm) * max(WIDTH, HEIGHT)
                    if dist_change > 0:  # Moving closer
                        reward += 0.03
            else:
                reward -= 0.02  # Penalty for being too far

        elif self.agent.role == 'healer':
            # Healers want to stay at medium distance, close to allies
            optimal_min, optimal_max = 80, 160
            if optimal_min <= current_dist <= optimal_max:
                reward += 0.03
            else:
                reward -= 0.01


        # Use a margin (pixels) near any wall
        margin = 100
        dist_left   = self.agent.x
        dist_right  = WIDTH - self.agent.x
        dist_top    = self.agent.y
        dist_bottom = HEIGHT - self.agent.y
        dist_to_wall = min(dist_left, dist_right, dist_top, dist_bottom)

        # Continuous penalty that ramps up near walls
        if dist_to_wall < margin:
            # Linearly scale from 0 at margin to -pen_max at the wall
            pen_max = 0.3
            wall_pen = -pen_max * (1.0 - dist_to_wall / margin)
            reward += wall_pen

        # Retreat penalty only when not dodging
        is_telegraphed = (self.boss.active_ability is not None)
        if self.prev_distance_norm is not None:
            dist_diff = current_dist - (self.prev_distance_norm * max(WIDTH, HEIGHT))
            if dist_diff > 2.0:  # moved away
                reward += 0.10 if is_telegraphed else -0.10

            
        # Set current as previous for next step
        self.prev_distance_norm = current_dist_norm

        self.prev_boss_health = self.boss.health
        self.prev_agent_health = self.agent.health

        self.num_steps += 1

        terminated = False
        truncated = False

        if self.boss.health <= 0:
            terminated = True
            reward += 40
        if self.agent.health <= 0:
            terminated = True
            if self.boss.health == self.boss.max_health:
                reward -=40
            reward -= 20
        if self.num_steps >= self.max_steps:
            truncated = True
            reward -=20

        obs = self._get_observation()
        info: Dict[str, Any] = {}

        return obs, reward, terminated, truncated, info

    # --- Observation and action helpers ---
    def _get_observation(self) -> np.ndarray:
        agent = self.agent
        boss = self.boss
        assert agent is not None and boss is not None

        agent_hp = agent.health / max(1, agent.max_health)
        agent_x = agent.x / WIDTH
        agent_y = agent.y / HEIGHT

        boss_hp = boss.health / max(1, boss.max_health)
        rel_x = (boss.x - agent.x) / WIDTH
        rel_y = (boss.y - agent.y) / HEIGHT
        dist = math.hypot(boss.x - agent.x, boss.y - agent.y)
        dist_norm = min(1.0, dist / max(WIDTH, HEIGHT))

        # Active ability compact encoding
        ability_id = 0.0
        ticks_norm = 0.0
        if boss.active_ability is not None:
            name = boss.active_ability.name
            # TODO: Consider one-hot ability encoding for better learnability
            name_to_id = {"Frontal Cone": 0, "Tank Buster": 1, "Fireball": 2}
            ability_id = (name_to_id.get(name, 0)) / 2.0  # normalized to [0,1]
            # Normalize by a conservative max windup (e.g., 40 ticks)
            ticks_norm = min(1.0, boss.ability_ticks_remaining / 20)

        step_progress = self.num_steps / max(1, self.max_steps)
        speed_norm = min(1.0, agent.speed / 5.0)

        obs = np.array([
            agent_hp, agent_x, agent_y,
            boss_hp, rel_x, rel_y, dist_norm,
            ability_id, ticks_norm,
            step_progress, speed_norm
        ], dtype=np.float32)
        return obs

    @staticmethod
    def _action_to_unit_vector(action: int) -> Tuple[float, float]:
        # 0: stay, 1-8: compass + diagonals (N, NE, E, SE, S, SW, W, NW)
        mapping = {
            0: (0.0, 0.0),
            1: (0.0, -1.0),
            2: (1.0, -1.0),
            3: (1.0, 0.0),
            4: (1.0, 1.0),
            5: (0.0, 1.0),
            6: (-1.0, 1.0),
            7: (-1.0, 0.0),
            8: (-1.0, -1.0),
        }
        dx, dy = mapping.get(action, (0.0, 0.0))
        # Normalize diagonal speed to keep consistent per-step distance before Agent.speed scaling
        length = math.hypot(dx, dy)
        if length > 0:
            dx /= length
            dy /= length
        return dx, dy

    # --- Rendering ---
    def render(self, mode: str = "human"):
        # Lazy import pygame to avoid training overhead
        import pygame  # noqa: WPS433

        if self.headless and mode == "human":
            # Headless mode requested human render: ignore silently
            return None

        if not self._pygame_init_done:
            pygame.init()
            self._screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Battle Arena Env")
            self._clock = pygame.time.Clock()
            try:
                pygame.font.init()
                self._font = pygame.font.SysFont("consolas", 16)
            except Exception:
                self._font = None
            self._pygame_init_done = True

        assert self._screen is not None
        screen = self._screen
        screen.fill((24, 24, 28))

        # Handle quit events lightly to keep window responsive during eval
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Optional: set a flag or simply close
                pygame.display.quit()
                self._pygame_init_done = False
                return None

        # Delegate drawing to existing classes
        if self.boss is not None:
            self.boss.draw(screen)
        if self.agent is not None:
            self.agent.draw(screen)

        # Overlays: health labels and training text
        if self._font is not None:
            if self.boss is not None:
                boss_hp_text = f"Boss HP: {int(self.boss.health)} / {int(self.boss.max_health)}"
                boss_surf = self._font.render(boss_hp_text, True, (240, 240, 240))
                screen.blit(boss_surf, (self.boss.x - boss_surf.get_width() // 2, self.boss.y + self.boss.radius + 6))
            if self.agent is not None:
                agent_hp_text = f"{self.agent.role.capitalize()} HP: {int(self.agent.health)} / {int(self.agent.max_health)}"
                agent_surf = self._font.render(agent_hp_text, True, (240, 240, 240))
                screen.blit(agent_surf, (self.agent.x - agent_surf.get_width() // 2, self.agent.y - self.agent.radius - 20))

            if self.overlay_text:
                overlay_surf = self._font.render(self.overlay_text, True, (200, 220, 255))
                screen.blit(overlay_surf, (10, 10))

        if self.action_history:
            action_names = [self._action_to_name(a) for a in self.action_history] 
            history_text = "Actions: " + " ".join(action_names)
            history_surf = self._font.render(history_text, True, (0,255,0))
            screen.blit(history_surf, (10,550))

        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(60)  # limit FPS during visualization

        if mode == "rgb_array":
            # Return an RGB array of the current frame
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(screen)), (1, 0, 2)
            ).copy()
        return None

    def close(self):
        if self._pygame_init_done:
            try:
                import pygame  # noqa: WPS433
                pygame.display.quit()
            except Exception:
                pass
        self._pygame_init_done = False

    def _action_to_name(self, action: int) -> str:
        action_names = {
            0: "x", 1: "↑", 2: "↓", 3: "←", 4: "→",
            5: "↖", 6: "↗", 7: "↙", 8: "↘", 9: "⚔"
        }
        return action_names.get(action, f"Unknown({action})")


# Convenience factory for gym.make users if desired
def make_env(**kwargs) -> BattleArenaEnv:
    return BattleArenaEnv(**kwargs)


