# env_alara.py — Environnement d'évacuation en zone irradiée (principe ALARA)


import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RadiationEvacuationEnv(gym.Env):
    """
    État observé (DQN)  : np.ndarray (2,) = (x/(N-1), y/(N-1)), float32 dans [0,1]
    État tabulaire      : entier state_id = x*N + y, fourni dans info
    Actions             : 0=haut, 1=bas, 2=gauche, 3=droite
    Récompense          : -lambda * (dose_pas / dose_max) - step_cost ; +goal_reward à la sortie
    Fin d'épisode       : terminated si sortie atteinte ; truncated si max_steps dépassé
    """

    metadata = {"render_modes": ["ansi"]}
    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}   # haut, bas, gauche, droite
    ACTION_NAMES = ["haut", "bas", "gauche", "droite"]

    def __init__(self, size=15, lambda_dose=1.0, step_cost=1.0,
                 goal_reward=100.0, max_steps=200):
        super().__init__()
        self.size = size
        self.lambda_dose = lambda_dose
        self.step_cost = step_cost
        self.goal_reward = goal_reward
        self.max_steps = max_steps

        # --- Espaces (interface : ne pas changer sans accord du groupe) ---
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        # --- Géométrie validée par le groupe ---
        self.start = (size // 2, 0)          # (7, 0)  milieu bord gauche
        self.goal = (size // 2, size - 1)    # (7, 14) milieu bord droit
        self.sources = [(7, 7, 60.0), (5, 8, 25.0), (9, 6, 25.0)]
        self.obstacles = {(2, 5), (3, 5), (11, 9), (12, 9)}

        # --- Physique : carte de dose pré-calculée (étape 1) ---
        self.dose_map = self._build_dose_map()
        self.dose_max = float(self.dose_map.max())

        # --- État interne ---
        self.pos = self.start
        self.t = 0
        self.cumulative_dose = 0.0

    # ------------------------------------------------------------ physique (étape 1)
    def _build_dose_map(self):
        """dose(p) = somme_k I_k / (distance(p, source_k)^2 + 1)."""
        dose = np.zeros((self.size, self.size), dtype=np.float32)
        for i in range(self.size):
            for j in range(self.size):
                total = 0.0
                for (si, sj, inten) in self.sources:
                    d2 = (i - si) ** 2 + (j - sj) ** 2
                    total += inten / (d2 + 1.0)
                dose[i, j] = total
        for o in self.obstacles:
            dose[o] = 0.0
        return dose

    # ------------------------------------------------------------ helpers
    def _obs(self):
        i, j = self.pos
        return np.array([i / (self.size - 1), j / (self.size - 1)], dtype=np.float32)

    def _state_id(self):
        return self.pos[0] * self.size + self.pos[1]

    def _info(self, dose_step=0.0):
        return {
            "pos": self.pos,
            "state_id": self._state_id(),
            "dose_step": float(dose_step),
            "cumulative_dose": float(self.cumulative_dose),
        }

    def _valid(self, pos):
        i, j = pos
        return 0 <= i < self.size and 0 <= j < self.size and pos not in self.obstacles

    # ------------------------------------------------------------ API gym (étape 2)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.start
        self.t = 0
        self.cumulative_dose = 0.0
        return self._obs(), self._info(0.0)

    def step(self, action):
        assert self.action_space.contains(action), f"action invalide : {action}"
        di, dj = self.ACTIONS[int(action)]
        new_pos = (self.pos[0] + di, self.pos[1] + dj)
        if self._valid(new_pos):
            self.pos = new_pos
        # sinon : mur/bord -> l'agent reste sur place mais paie le pas

        dose_step = float(self.dose_map[self.pos])
        self.cumulative_dose += dose_step
        self.t += 1

        reward = -self.lambda_dose * (dose_step / self.dose_max) - self.step_cost
        terminated, truncated = False, False
        if self.pos == self.goal:
            reward += self.goal_reward
            terminated = True
        elif self.t >= self.max_steps:
            truncated = True

        return self._obs(), reward, terminated, truncated, self._info(dose_step)

    def render(self):
        chars = np.full((self.size, self.size), ".", dtype="<U2")
        for o in self.obstacles:
            chars[o] = "#"
        for (si, sj, _) in self.sources:
            chars[si, sj] = "S"
        chars[self.goal] = "G"
        chars[self.pos] = "A"
        return "\n".join(" ".join(row) for row in chars)


# ---------------------------------------------------------------------------
# Vérification rapide de l'interface (python env_alara.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = RadiationEvacuationEnv()
    obs, info = env.reset()
    assert obs.shape == (2,) and "state_id" in info
    total_r = 0.0
    for _ in range(30):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total_r += r
        if term or trunc:
            break
    print("Interface OK.")
    print(f"  grille {env.size}x{env.size} | départ {env.start} -> sortie {env.goal}")
    print(f"  dose_max = {env.dose_max:.1f} µSv/h")
    print(f"  obs = {obs} (forme {obs.shape}) | state_id = {info['state_id']}")
    print(f"  récompense cumulée sur l'essai aléatoire = {total_r:.2f}")
