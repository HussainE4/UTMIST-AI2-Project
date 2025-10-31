import multiprocessing as mp
from functools import partial
import os
import time




# Worker function to train a single agent
def train_agent_proc(agent_idx: int, pretrain_timesteps: int, save_path: str):
    # --- import inside function for Windows multiprocessing ---
    from environment.agent import RecurrentPPOAgent, train, SelfPlayRandom
    from environment.agent import OpponentsCfg, SaveHandler, SaveHandlerMode
    from environment.agent import CameraResolution, TrainLogging
    from user.train_agent import gen_reward_manager

    reward_manager = gen_reward_manager()
    agent = RecurrentPPOAgent()

    save_handler = SaveHandler(
        agent=agent,
        save_freq=20_000,
        max_saved=10,
        save_path=save_path,
        run_name=f"agent_{agent_idx}",
        mode=SaveHandlerMode.FORCE
    )

    # Initially, no opponents except self-play handler
    selfplay_handler = SelfPlayRandom(partial(RecurrentPPOAgent))
    opponent_cfg = OpponentsCfg(opponents={'self_play': (8, selfplay_handler)})

    train(agent, reward_manager, save_handler, opponent_cfg,
          CameraResolution.LOW,
          train_timesteps=pretrain_timesteps,
          train_logging=TrainLogging.PLOT)

    # Return latest checkpoint path
    return save_handler.latest_checkpoint()

def train_agent_worker(agent_idx: int, pretrain_timesteps: int, save_path: str, queue: mp.Queue):
    ckpt = train_agent_proc(agent_idx, pretrain_timesteps, save_path)
    queue.put((agent_idx, ckpt))

def run_selfplay_round(num_agents=3, pretrain_timesteps=100_000, save_path="checkpoints"):
    os.makedirs(save_path, exist_ok=True)

    processes = []
    results_queue = mp.Queue()

    for i in range(num_agents):
        p = mp.Process(target=train_agent_worker, args=(i, pretrain_timesteps, save_path, results_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    checkpoints = [None] * num_agents
    while not results_queue.empty():
        idx, ckpt = results_queue.get()
        checkpoints[idx] = ckpt

    return checkpoints

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Windows-safe
    num_rounds = 5
    num_agents = 3
    save_path = "checkpoints"

    # Keep track of checkpoints across rounds
    last_checkpoints = [None] * num_agents

    for round_idx in range(num_rounds):
        print(f"--- Self-play round {round_idx + 1} ---")
        agents, opp_cfgs, last_checkpoints = run_selfplay_round(num_agents=num_agents,
                                                                pretrain_timesteps=5_000,
                                                                save_path=save_path)
        # Optionally, you could do further fine-tuning here
        print("Checkpoints updated:", last_checkpoints)
