from typing import Optional


import gymnasium
from stable_baselines3.common.callbacks import BaseCallback

from environment.environment import WarehouseBrawl
from user.Common import OpponentsCfg, RewardManager, SaveHandler, Agent
from environment.environment import CameraResolution


class SelfPlayWarehouseBrawl(gymnasium.Env):
    """SubprocVecEnv-safe version of the environment."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 opponent_cfg: Optional[OpponentsCfg] = None,
                 render_every: int | None = None,
                 resolution: CameraResolution = CameraResolution.LOW):
        """
        Args:
            opponent_cfg (OpponentsCfg): Configuration for opponents.
            render_every (int | None): Steps between demo renders (None for offscreen).
        """
        super().__init__()

        self.opponent_cfg = opponent_cfg or OpponentsCfg()
        self.render_every = render_every
        self.resolution = resolution

        # Lazy-loaded attributes
        self.reward_manager = None
        self.save_handler = None
        self.raw_env = None
        self.opponent_agent = None
        self.opponent_obs = None
        self.games_done = 0

        # Predefine dummy spaces so VecEnv can inspect
        dummy_env = WarehouseBrawl(resolution=resolution, train_mode=True)
        self.action_space = dummy_env.action_space
        self.observation_space = dummy_env.observation_space

        # Provide a minimal obs_helper so agent can inspect env
        self.obs_helper = dummy_env.obs_helper
        self.act_helper = dummy_env.act_helper
        del dummy_env

    # --- Lazy init helpers -------------------------------------------------------

    def _lazy_init(self):
        """Create heavy objects when environment actually starts."""
        if self.raw_env is not None:
            return

        self.raw_env = WarehouseBrawl(resolution=self.resolution, train_mode=True)

        if self.reward_manager is None and self.reward_manager_cfg:
            self.reward_manager = RewardManager(**self.reward_manager_cfg)
            self.reward_manager.subscribe_signals(self.raw_env)

        if self.save_handler is None and self.save_handler_cfg:
            self.save_handler = SaveHandler(**self.save_handler_cfg)

        # Avoid setting circular refs here — handled per-episode if needed.
        self.opponent_cfg.validate_probabilities()

    # --- Gym API ----------------------------------------------------------------

    def reset(self, seed=None, options=None):
        self._lazy_init()

        observations, info = self.raw_env.reset()
        if self.reward_manager:
            self.reward_manager.reset()

        new_agent = self.opponent_cfg.on_env_reset()
        if new_agent is not None:
            self.opponent_agent = new_agent
        self.opponent_obs = observations[1]

        self.games_done += 1
        return observations[0], info

    def step(self, action):
        full_action = {
            0: action,
            1: self.opponent_agent.predict(self.opponent_obs) if self.opponent_agent else self.action_space.sample()
        }

        observations, rewards, terminated, truncated, info = self.raw_env.step(full_action)
        self.opponent_obs = observations[1]

        if self.reward_manager:
            reward = self.reward_manager.process(self.raw_env, 1 / 30.0)
        else:
            reward = rewards[0]

        if self.save_handler:
            self.save_handler.process()

        return observations[0], reward, terminated, truncated, info

    def render(self):
        if self.raw_env:
            return self.raw_env.render()

    def close(self):
        if self.raw_env:
            self.raw_env.close()
            self.raw_env = None

    def on_training_start(self):
        """Hook called at the start of training."""
        if self.save_handler is not None:
            self.save_handler.update_info()

    def on_training_end(self):
        """Hook called at the end of training."""
        if self.save_handler is not None:
            self.save_handler.agent.update_num_timesteps(self.save_handler.num_timesteps)
            self.save_handler.save_agent()

    def reset(self, seed=None, options=None):
        # Reset MalachiteEnv
        observations, info = self.raw_env.reset()

        self.reward_manager.reset()

        # Select agent
        new_agent: Agent = self.opponent_cfg.on_env_reset()
        if new_agent is not None:
            self.opponent_agent: Agent = new_agent
        self.opponent_obs = observations[1]


        self.games_done += 1
        #if self.games_done % self.render_every == 0:
            #self.render_out_video()

        return observations[0], info

    def render(self):
        img = self.raw_env.render()
        return img

    def close(self):
        pass

