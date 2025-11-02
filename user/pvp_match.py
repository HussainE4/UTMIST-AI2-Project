from environment.environment import RenderMode, CameraResolution
from environment.agent import run_real_time_match
from user.train_agent import UserInputAgent, BasedAgent, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent, CustomAgent #add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
import pygame
pygame.init()


# my_agent = CustomAgent(file_path="checkpoints/scared_new_agent/rl_model_1004400_steps.zip")
# my_agent = CustomAgent(file_path="checkpoints/balanced_new_agent/clone_yuj_mlp_first_v12")
# my_agent = CustomAgent(file_path="checkpoints/balanced_new_agent/rl_model_1004400_steps.zip")
# my_agent = BasedAgent()
# my_agent = UserInputAgent()
my_agent = CustomAgent(file_path="checkpoints/based_starter/rl_model_0_steps.zip")
# my_agent = SubmittedAgent()
#Input your file path here in SubmittedAgent if you are loading a model:
# opponent = BasedAgent()
opponent = ConstantAgent()

match_time = 99999

# Run a single real-time match
run_real_time_match(
    agent_1=my_agent,
    agent_2=opponent,
    max_timesteps=30 * 999990000,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
)