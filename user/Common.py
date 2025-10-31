import os
import random
import shutil
from abc import abstractmethod, ABC
from dataclasses import field, dataclass, MISSING
from enum import Enum
from functools import partial
from typing import Optional, Any, List, Dict, Tuple, Callable

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

class Agent(ABC):

    def __init__(
            self,
            file_path: Optional[str] = None
        ):

        # If no supplied file_path, load from gdown (optional file_path returned)
        if file_path is None:
            file_path = self._gdown()

        self.file_path: Optional[str] = file_path
        self.initialized = False

    def get_env_info(self, env, real_env = None):
        if (real_env is None):
            real_env = env
        if isinstance(env, Monitor):
            self_env = env.env
        else:
            self_env = env
        self.observation_space = self_env.observation_space
        self.obs_helper = self_env.obs_helper
        self.action_space = self_env.action_space
        self.act_helper = self_env.act_helper
        self.env = real_env
        self._initialize()
        self.initialized = True

    def get_num_timesteps(self) -> int:
        if hasattr(self, 'model'):
            return self.model.num_timesteps
        else:
            return 0

    def update_num_timesteps(self, num_timesteps: int) -> None:
        if hasattr(self, 'model'):
            self.model.num_timesteps = num_timesteps

    @abstractmethod
    def predict(self, obs) -> spaces.Space:
        pass

    def save(self, file_path: str) -> None:
        return

    def reset(self) -> None:
        return

    def _initialize(self) -> None:
        """

        """
        return

    def _gdown(self) -> Optional[str]:
        """
        Loads the necessary file from Google Drive, returning a file path.
        Or, returns None, if the agent does not require loaded files.

        :return:
        """
        return
class ConstantAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = np.zeros_like(self.action_space.sample())
        return action

class RandomAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.action_space.sample()
        return action

@dataclass
class OpponentsCfg():
    """Configuration for opponents.

    Args:
        swap_steps (int): Number of steps between swapping opponents.
        opponents (dict): Dictionary specifying available opponents and their selection probabilities.
    """
    swap_steps: int = 10_000
    opponents: dict[str, Any] = field(default_factory=lambda: {
                'random_agent': (0.8, partial(RandomAgent)),
                'constant_agent': (0.2, partial(ConstantAgent)),
                #'recurrent_agent': (0.1, partial(RecurrentPPOAgent, file_path='skibidi')),
            })

    def validate_probabilities(self) -> None:
        total_prob = sum(prob if isinstance(prob, float) else prob[0] for prob in self.opponents.values())

        if abs(total_prob - 1.0) > 1e-5:
            print(f"Warning: Probabilities do not sum to 1 (current sum = {total_prob}). Normalizing...")
            self.opponents = {
                key: (value / total_prob if isinstance(value, float) else (value[0] / total_prob, value[1]))
                for key, value in self.opponents.items()
            }

    def process(self) -> None:
        pass

    def on_env_reset(self) -> Agent:

        agent_name = random.choices(
            list(self.opponents.keys()),
            weights=[prob if isinstance(prob, float) else prob[0] for prob in self.opponents.values()]
        )[0]

        # If self-play is selected, return the trained model
        print(f'Selected {agent_name}')
        if agent_name == "self_play":
            selfplay_handler: SelfPlayHandler = self.opponents[agent_name][1]
            return selfplay_handler.get_opponent(self.env)
        else:
            # Otherwise, return an instance of the selected agent class
            opponent = self.opponents[agent_name][1]()

        opponent.get_env_info(self.env)
        return opponent


class SelfPlayHandler(ABC):
    """Handles self-play."""

    def __init__(self, agent_partial: partial):
        self.agent_partial = agent_partial

    def get_model_from_path(self, path, env) -> Agent:
        if path:
            try:
                opponent = self.agent_partial(file_path=path)
            except FileNotFoundError:
                print(f"Warning: Self-play file {path} not found. Defaulting to constant agent.")
                opponent = ConstantAgent()
        else:
            print("Warning: No self-play model saved. Defaulting to constant agent.")
            opponent = ConstantAgent()
        opponent.get_env_info(env)
        return opponent

    @abstractmethod
    def get_opponent(self, env) -> Agent:
        pass

@dataclass
class RewTerm():
    """Configuration for a reward term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the reward signals as torch float tensors of
    shape (num_envs,).
    """

    weight: float = MISSING
    """The weight of the reward term.

    This is multiplied with the reward term's value to compute the final
    reward.

    Note:
        If the weight is zero, the reward term is ignored.
    """

    params: dict[str, Any] = field(default_factory=dict)
    """The parameters to be passed to the function as keyword arguments. Defaults to an empty dict.

    .. note::
        If the value is a :class:`SceneEntityCfg` object, the manager will query the scene entity
        from the :class:`InteractiveScene` and process the entity's joints and bodies as specified
        in the :class:`SceneEntityCfg` object.
    """

class RewardManager():
    """Reward terms for the MDP."""

    # (1) Constant running reward
    def __init__(self,
                 reward_functions: Optional[Dict[str, RewTerm]]=None,
                 signal_subscriptions: Optional[Dict[str, Tuple[str, RewTerm]]]=None) -> None:
        self.reward_functions = reward_functions
        self.signal_subscriptions = signal_subscriptions
        self.total_reward = 0.0
        self.collected_signal_rewards = 0.0

    def subscribe_signals(self, env) -> None:
        if self.signal_subscriptions is None:
            return
        for _, (name, term_cfg) in self.signal_subscriptions.items():
            getattr(env, name).connect(partial(self._signal_func, term_cfg))

    def _signal_func(self, term_cfg: RewTerm, *args, **kwargs):
        term_partial = partial(term_cfg.func, **term_cfg.params)
        self.collected_signal_rewards += term_partial(*args, **kwargs) * term_cfg.weight

    def process(self, env, dt) -> float:
        # reset computation
        reward_buffer = 0.0
        # iterate over all the reward terms
        if self.reward_functions is not None:
            for name, term_cfg in self.reward_functions.items():
                # skip if weight is zero (kind of a micro-optimization)
                if term_cfg.weight == 0.0:
                    continue
                # compute term's value
                value = term_cfg.func(env, **term_cfg.params) * term_cfg.weight
                # update total reward
                reward_buffer += value

        reward = reward_buffer + self.collected_signal_rewards
        self.collected_signal_rewards = 0.0

        self.total_reward += reward

        log = env.logger[0]
        log['reward'] = f'{reward_buffer:.3f}'
        log['total_reward'] = f'{self.total_reward:.3f}'
        env.logger[0] = log
        return reward

    def reset(self):
        self.total_reward = 0
        self.collected_signal_rewards


class SaveHandlerMode(Enum):
    FORCE = 0
    RESUME = 1

class SaveHandler():
    """Handles saving.

    Args:
        agent (Agent): Agent to save.
        save_freq (int): Number of steps between saving.
        max_saved (int): Maximum number of saved models.
        save_dir (str): Directory to save models.
        name_prefix (str): Prefix for saved models.
    """

    # System for saving to internet

    def __init__(
            self,
            agent: Agent,
            save_freq: int=10_000,
            max_saved: int=20,
            run_name: str='experiment_1',
            save_path: str='checkpoints',
            name_prefix: str = "rl_model",
            mode: SaveHandlerMode=SaveHandlerMode.FORCE
        ):
        self.agent = agent
        self.save_freq = save_freq
        self.run_name = run_name
        self.max_saved = max_saved
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.mode = mode

        self.steps_until_save = save_freq
        # Get model paths from exp_path, if it exists
        exp_path = self._experiment_path()
        self.history: List[str] = []
        if self.mode == SaveHandlerMode.FORCE:
            # Clear old dir
            if os.path.exists(exp_path) and len(os.listdir(exp_path)) != 0:
                while True:
                    answer = "y"
                    if answer in ('y', 'n'):
                        break
                    else:
                        print("Invalid input, please enter 'y' or 'n'.")

                if answer == 'n':
                    raise ValueError('Please switch to SaveHandlerMode.FORCE or use a new run_name.')
                print(f'Clearing {exp_path}...')
                if os.path.exists(exp_path):
                    shutil.rmtree(exp_path)
            else:
                print(f'{exp_path} empty or does not exist. Creating...')

            if not os.path.exists(exp_path):
                os.makedirs(exp_path)
        elif self.mode == SaveHandlerMode.RESUME:
            self.update_history(exp_path, max_saved)

    def update_history(self, exp_path, max_saved):
        if os.path.exists(exp_path):
            # Get all model paths
            self.history = [os.path.join(exp_path, f) for f in os.listdir(exp_path) if
                            os.path.isfile(os.path.join(exp_path, f))]
            # Filter any non .csv
            self.history = [f for f in self.history if f.endswith('.zip')]
            if len(self.history) != 0:
                self.history.sort(key=lambda x: int(os.path.basename(x).split('_')[-2].split('.')[0]))
                if max_saved != -1: self.history = self.history[-max_saved:]
                print(f'Best model is {self.history[-1]}')
            else:
                print(f'No models found in {exp_path}.')
        else:
            print(f'No file found at {exp_path}')

    def update_info(self) -> None:
        self.num_timesteps = self.agent.get_num_timesteps()

    def _experiment_path(self) -> str:
        """
        Helper to get experiment path for each type of checkpoint.

        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, self.run_name)

    def _checkpoint_path(self, extension: str = '') -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self._experiment_path(), f"{self.name_prefix}_{self.num_timesteps}_steps.{extension}")

    def save_agent(self) -> None:
        print(f"Saving agent to {self._checkpoint_path()}")
        model_path = self._checkpoint_path('zip')
        self.agent.save(model_path)
        self.history.append(model_path)
        if self.max_saved != -1 and len(self.history) > self.max_saved:
            os.remove(self.history.pop(0))

    def process(self) -> bool:
        self.num_timesteps += 1

        if self.steps_until_save <= 0:
            # Save agent
            self.steps_until_save = self.save_freq
            self.save_agent()
            return True
        self.steps_until_save -= 1

        return False

    def get_random_model_path(self) -> str:
        print(self._experiment_path())
        self.update_history(self._experiment_path(), self.max_saved)
        if len(self.history) == 0:
            return None
        return random.choice(self.history)

    def get_latest_model_path(self) -> str:
        self.update_history(self._experiment_path(), self.max_saved)
        if len(self.history) == 0:
            return None
        return self.history[-1]

class SaveHandlerCallback(BaseCallback):
    def __init__(self, save_handler, save_freq, scale):
        super().__init__(verbose=1)
        self.save_handler = save_handler
        self.save_freq = save_freq
        self.scale = scale

    def _on_step(self):
        for i in range(self.scale):
            self.save_handler.process()
        return True