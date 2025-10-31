from stable_baselines3.common.logger import configure

from environment.brawl import SelfPlayWarehouseBrawl
from environment.environment import ActHelper, AirTurnaroundState, Animation, AnimationSprite2D, AttackState, BackDashState, Camera, CameraResolution, Capsule, CapsuleCollider, Cast, CastFrameChangeHolder, CasterPositionChange, CasterVelocityDampXY, CasterVelocitySet, CasterVelocitySetXY, CompactMoveState, DashState, DealtPositionTarget, DodgeState, Facing, GameObject, Ground, GroundState, HurtboxPositionChange, InAirState, KOState, KeyIconPanel, KeyStatus, MalachiteEnv, MatchStats, MoveManager, MoveType, ObsHelper, Particle, Player, PlayerInputHandler, PlayerObjectState, PlayerStats, Power, RenderMode, Result, Signal, SprintingState, Stage, StandingState, StunState, Target, TauntState, TurnaroundState, UIHandler, WalkingState, WarehouseBrawl, hex_to_rgb

import warnings
from typing import TYPE_CHECKING, Any, Generic, \
 SupportsFloat, TypeVar, Type, Optional, List, Dict, Callable
from enum import Enum, auto
from functools import partial
from typing import Tuple, Any

from PIL import Image, ImageSequence
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn

import pygame.gfxdraw

import skvideo
import skvideo.io

from user.Common import Agent, SelfPlayHandler, SaveHandler, SaveHandlerMode, OpponentsCfg, RewardManager, \
    SaveHandlerCallback
from user.agents import CustomAgent, BigMLPExtractor

# ## Agents

# ### Agent Abstract Base Class

# In[ ]:


SelfAgent = TypeVar("SelfAgent", bound="Agent")


# ### Agent Classes

# In[ ]:


class ConstantAgent(Agent):
    '''
    ConstantAgent:
    - The ConstantAgent simply is in an IdleState (action_space all equal to zero.)
    As such it will not do anything, DON'T use this agent for your training.
    '''
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
    '''
    RandomAgent:
    - The RandomAgent (as it name says) simply samples random actions.
    NOT used for training
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.action_space.sample()
        return action


# ## StableBaselines3 Integration

# ### Reward Configuration

# In[ ]:





# ### Save, Self-play, and Opponents

# In[ ]:



class SelfPlayLatest(SelfPlayHandler):
    def __init__(self, agent_partial: partial):
        super().__init__(agent_partial)
    
    def get_opponent(self, env) -> Agent:
        assert self.save_handler is not None, "Save handler must be specified for self-play"
        chosen_path = self.save_handler.get_latest_model_path()
        return self.get_model_from_path(chosen_path, env)

class SelfPlayRandom(SelfPlayHandler):
    def __init__(self, agent_partial: partial):
        super().__init__(agent_partial)
    
    def get_opponent(self, env) -> Agent:
        assert self.save_handler is not None, "Save handler must be specified for self-play"
        chosen_path = self.save_handler.get_random_model_path()
        return self.get_model_from_path(chosen_path, env)


# ## Run Match

# In[ ]:


from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize
from tqdm import tqdm

def run_match(agent_1: Agent | partial,
              agent_2: Agent | partial,
              max_timesteps=30*90,
              video_path: Optional[str]=None,
              agent_1_name: Optional[str]=None,
              agent_2_name: Optional[str]=None,
              resolution = CameraResolution.LOW,
              reward_manager: Optional[RewardManager]=None,
              train_mode=False
              ) -> MatchStats:
    # Initialize env

    env = WarehouseBrawl(resolution=resolution, train_mode=train_mode)
    observations, infos = env.reset()
    obs_1 = observations[0]
    obs_2 = observations[1]
    print("RUN MATCH IS RUNNING")
    if reward_manager is not None:
        reward_manager.reset()
        reward_manager.subscribe_signals(env)

    if agent_1_name is None:
        agent_1_name = 'agent_1'
    if agent_2_name is None:
        agent_2_name = 'agent_2'

    env.agent_1_name = agent_1_name
    env.agent_2_name = agent_2_name


    writer = None
    if video_path is None:
        print("video_path=None -> Not rendering")
    else:
        print(f"video_path={video_path} -> Rendering")
        # Initialize video writer
        writer = skvideo.io.FFmpegWriter(video_path, outputdict={
            '-vcodec': 'libx264',  # Use H.264 for Windows Media Player
            '-pix_fmt': 'yuv420p',  # Compatible with both WMP & Colab
            '-preset': 'fast',  # Faster encoding
            '-crf': '20',  # Quality-based encoding (lower = better quality)
            '-r': '30'  # Frame rate
        })

    # If partial
    if callable(agent_1):
        agent_1 = agent_1()
    if callable(agent_2):
        agent_2 = agent_2()

    # Initialize agents
    if not agent_1.initialized: agent_1.get_env_info(env)
    if not agent_2.initialized: agent_2.get_env_info(env)
    # 596, 336
    platform1 = env.objects["platform1"]

    for time in tqdm(range(max_timesteps), total=max_timesteps):
      platform1.physics_process(0.05)
      full_action = {
          0: agent_1.predict(obs_1),
          1: agent_2.predict(obs_2)
      }

      observations, rewards, terminated, truncated, info = env.step(full_action)
      obs_1 = observations[0]
      obs_2 = observations[1]

      if reward_manager is not None:
          reward_manager.process(env, 1 / env.fps)

      if video_path is not None:
            img = env.render()
            img = np.rot90(img, k=-1)  #video output rotate fix
            img = np.fliplr(img)  # Mirror/flip the image horizontally
            writer.writeFrame(img) 
            del img

      if terminated or truncated:
          break


    if video_path is not None:
        writer.close()
    env.close()

    # visualize
    # Video(video_path, embed=True, width=800) if video_path is not None else None
    player_1_stats = env.get_stats(0)
    player_2_stats = env.get_stats(1)

    if player_1_stats.lives_left > player_2_stats.lives_left:
        result = Result.WIN
    elif player_1_stats.lives_left < player_2_stats.lives_left:
        result = Result.LOSS
    else:
        result = Result.DRAW
    
    match_stats = MatchStats(
        match_time=env.steps / env.fps,
        player1=player_1_stats,
        player2=player_2_stats,
        player1_result=result
    )

    del env

    return match_stats





class BasedAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class UserInputAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()
       
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)

        #if keys[pygame.K_q]:
        #    action = self.act_helper.press_keys(['q'], action)
        #if keys[pygame.K_v]:
        #    action = self.act_helper.press_keys(['v'], action)
        return action


class ClockworkAgent(Agent):

    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.current_action_end = 0  # Tracks when the current action should stop
        self.current_action_data = None  # Stores the active action
        self.action_index = 0  # Index in the action sheet

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (30, []),
                (7, ['d']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (1, ['a']),
                (4, ['a','l']),
                (20, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
            ]
        else:
            self.action_sheet = action_sheet


    def predict(self, obs):
        """
        Returns an action vector based on the predefined action sheet.
        """
        # Check if the current action has expired
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data  # Store the action
            self.current_action_end = self.steps + hold_time  # Set duration
            self.action_index += 1  # Move to the next action

        # Apply the currently active action
        action = self.act_helper.press_keys(self.current_action_data)


        self.steps += 1  # Increment step counter
        return action

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

class SB3Agent(Agent):

    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

from sb3_contrib import RecurrentPPO

class RecurrentPPOAgent(Agent):

    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 512,
                'net_arch': [dict(pi=[32, 32], vf=[32, 32])],
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True,

            }
            self.model = RecurrentPPO("MlpLstmPolicy",
                                      self.env,
                                      verbose=0,
                                      n_steps=30*90*20,
                                      batch_size=16,
                                      ent_coef=0.05,
                                      policy_kwargs=policy_kwargs)
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path)

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)


# ## Training Function
# A helper function for training.

# In[ ]:


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

class TrainLogging(Enum):
    NONE = 0
    TO_FILE = 1
    PLOT = 2

def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")

    weights = np.repeat(1.0, 50) / 50
    print(weights, y)
    y = np.convolve(y, weights, "valid")
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")

    # save to file
    plt.savefig(log_folder + title + ".png")

def make_env(agent, rank, reward_manager, opponent_cfg, resolution, path):
    def _init():

        env = SelfPlayWarehouseBrawl(
            opponent_cfg=opponent_cfg,
            resolution=resolution
        )

        env.reward_manager = reward_manager
        env.raw_env = WarehouseBrawl(resolution=resolution, train_mode=True)
        env.opponent_obs = None
        env.opponent_agent = None
        env.opponent_cfg.env = env
        env.opponent_cfg.validate_probabilities()

        log_dir = f"{path}_{rank}"
        os.makedirs(os.path.join('checkpoints', path), exist_ok=True)
        env = Monitor(env, log_dir)
        return env
    return _init

def train(name,
          reward_manager: RewardManager,
          opponent_cfg: OpponentsCfg = OpponentsCfg(),
          resolution: CameraResolution = CameraResolution.LOW,
          train_timesteps: int = 400_000,
          train_logging: TrainLogging = TrainLogging.PLOT,
          n_envs: int = 8):
    """
    Parallel training using SubprocVecEnv while maintaining compatibility with single-thread training.
    """
    exp_path = os.path.join('checkpoints', name)
    # === Environment Factory ===


    agent = CustomAgent(sb3_class=PPO, extractor=BigMLPExtractor)

    save_handler = SaveHandler(
        agent=agent,
        save_freq=100_000,
        max_saved=400,
        save_path='checkpoints',
        run_name=name,
        mode=SaveHandlerMode.RESUME
    )

    for key, value in opponent_cfg.opponents.items():
        if isinstance(value[1], SelfPlayHandler):
            value[1].save_handler = save_handler

    single_env = SelfPlayWarehouseBrawl(
        opponent_cfg=opponent_cfg,
        resolution=resolution
    )
    # === Create VecEnv ===
    if n_envs > 1:
        env = SubprocVecEnv([make_env(agent, i, reward_manager, opponent_cfg, resolution, exp_path) for i in range(n_envs)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    else:
        env = make_env(agent, 0, reward_manager, opponent_cfg, resolution, exp_path)()

    # === Begin Training ===
    try:
        policy_kwargs = dict(
            net_arch=[64, 64],
            activation_fn=torch.nn.ReLU,
            ortho_init=True
        )

        agent.model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=1024,#30 * 90 * 3,
            learning_rate=0.001,
            batch_size=64,
            device="cuda",
        )

        save_handler.update_info()
        callback = SaveHandlerCallback(save_handler, save_freq=100_000, scale=n_envs)

        agent.get_env_info(single_env, env)
        single_env.close()
        agent.env = env
        agent.self_env = env

        if n_envs > 1:
            env.env_method("on_training_start")
        else:
            env.env.on_training_start()

        agent.learn(env,
                    total_timesteps=train_timesteps,
                    verbose=1,
                    callback=callback,
                    path=exp_path,
                    lr=0.001,
                    name=name)

        if n_envs > 1:
            env.env_method("on_training_end")
        else:
            env.env.on_training_end()


    except KeyboardInterrupt:
        print("⏹ Training interrupted manually.")
        if save_handler is not None:
            save_handler.agent.update_num_timesteps(save_handler.num_timesteps)
            save_handler.save_agent()

    finally:
        env.close()

    # === Save final model ===
    if save_handler is not None:
        save_handler.save_agent()



import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

## Run Human vs AI match function
import pygame
from pygame.locals import QUIT

def run_real_time_match(agent_1: UserInputAgent, agent_2: Agent, max_timesteps=30*90, resolution=CameraResolution.LOW):
    pygame.init()

    pygame.mixer.init()

    # Load your soundtrack (must be .wav, .ogg, or supported format)
    pygame.mixer.music.load("environment/assets/soundtrack.mp3")

    # Play it on loop: -1 = loop forever
    pygame.mixer.music.play(-1)

    # Optional: set volume (0.0 to 1.0)
    pygame.mixer.music.set_volume(0.2)

    resolutions = {
        CameraResolution.LOW: (480, 720),
        CameraResolution.MEDIUM: (720, 1280),
        CameraResolution.HIGH: (1080, 1920)
    }
    
    screen = pygame.display.set_mode(resolutions[resolution][::-1])  # Set screen dimensions


    pygame.display.set_caption("AI Squared - Player vs AI Demo")

    clock = pygame.time.Clock()

    # Initialize environment
    env = WarehouseBrawl(resolution=resolution, train_mode=False)
    observations, _ = env.reset()
    obs_1 = observations[0]
    obs_2 = observations[1]

    if not agent_1.initialized: agent_1.get_env_info(env)
    if not agent_2.initialized: agent_2.get_env_info(env)

    # Run the match loop
    running = True
    timestep = 0
   # platform1 = env.objects["platform1"] #mohamed
    #stage2 = env.objects["stage2"]
    background_image = pygame.image.load('environment/assets/map/bg.jpg').convert() 
    while running and timestep < max_timesteps:
       
        # Pygame event to handle real-time user input 
       
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == pygame.VIDEORESIZE:
                 screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
       
        action_1 = agent_1.predict(obs_1)

        # AI input
        action_2 = agent_2.predict(obs_2)

        # Sample action space
        full_action = {0: action_1, 1: action_2}
        observations, rewards, terminated, truncated, info = env.step(full_action)
        obs_1 = observations[0]
        obs_2 = observations[1]

        # Render the game
        
        img = env.render()
        screen.blit(pygame.surfarray.make_surface(img), (0, 0))
     
        pygame.display.flip()

        # Control frame rate (30 fps)
        clock.tick(30)

        # If the match is over (either terminated or truncated), stop the loop
        if terminated or truncated:
            running = False

        timestep += 1

    # Clean up pygame after match
    pygame.quit()

    # Return match stats
    player_1_stats = env.get_stats(0)
    player_2_stats = env.get_stats(1)

    if player_1_stats.lives_left > player_2_stats.lives_left:
        result = Result.WIN
    elif player_1_stats.lives_left < player_2_stats.lives_left:
        result = Result.LOSS
    else:
        result = Result.DRAW
    
    match_stats = MatchStats(
        match_time=timestep / 30.0,
        player1=player_1_stats,
        player2=player_2_stats,
        player1_result=result
    )

    # Close environment
    env.close()

    return match_stats
