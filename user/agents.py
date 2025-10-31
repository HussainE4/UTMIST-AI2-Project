import os
from typing import Optional, Type
from torch.nn import functional as F
import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from user.Common import Agent

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        """
        A 3-layer MLP policy:
        obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
        """
        super(MLPPolicy, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)

    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        returns: [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class BigMLPExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for SB3 using the BigMLPPolicy above.
    Suitable for high-dimensional fighting game observations.
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 16, hidden_dim: int = 64):
        super(BigMLPExtractor, self).__init__(observation_space, features_dim)
        obs_dim = int(np.prod(observation_space.shape))
        self.model = MLPPolicy(
            obs_dim=obs_dim,
            action_dim=features_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)

    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 512, hidden_dim: int = 512) -> dict:
        """
        Helper to plug directly into PPO or RecurrentPPO agents:
            policy_kwargs = BigMLPExtractor.get_policy_kwargs()
        """
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim)
        )


class CustomAgent(Agent):
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]] = PPO, file_path: str = None,
                 extractor: BaseFeaturesExtractor = None):
        self.sb3_class = sb3_class
        self.extractor = extractor
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            # self.model = self.sb3_class("MlpPolicy", self.env, policy_kwargs=self.extractor.get_policy_kwargs(),
            #                             verbose=0, n_steps=30 * 90 * 3, batch_size=128, ent_coef=0.01, device="cuda")
            # del self.env
            pass
        else:
            self.model = self.sb3_class.load(self.file_path)


    def _gdown(self) -> str:
        # Call gdown to your link
        return

    # def set_ignore_grad(self) -> None:
    # self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0, callback=None, path=None, lr = 0.001, name=''):
        self.model.set_env(env)
        if path is not None:
            new_logger = configure(os.path.join("./tb_log", name), ["stdout", "tensorboard"])
            self.model.set_logger(new_logger)
        self.model.verbose = verbose
        self.model.learning_rate = lr
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            callback=callback
        )