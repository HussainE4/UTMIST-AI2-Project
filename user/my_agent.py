# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
#
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
#
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
#
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!
import math
import os
from enum import IntEnum

import gdown
from typing import Optional

import numpy as np
import torch
from torch import nn

from environment.agent import Agent

# To run the sample TTNN model, you can uncomment the 2 lines below:
import ttnn
#from user.my_agent_tt import TTMLPPolicy

class MovePrio(IntEnum):
    ATTACK_FIRST = 0
    ATTACK_OVERRIDE = 1
    CRITICAL = 2

class MoveType(IntEnum):
    NLIGHT = 2  # grounded light neutral
    DLIGHT = 3  # grounded light down
    SLIGHT = 4  # grounded light side
    NSIG = 5  # grounded heavy neutral
    DSIG = 6  # grounded heavy down
    SSIG = 7  # grounded heavy side
    NAIR = 8  # aerial light neutral
    DAIR = 9  # aerial light down
    SAIR = 10  # aerial light side
    RECOVERY = 11  # aerial heavy neutral and aerial heavy side
    GROUNDPOUND = 12  # aerial heavy down

    @staticmethod
    def prio_dodge():
        return [MoveType.GROUNDPOUND]

    @staticmethod
    def is_up(move):
        return move in [MoveType.NLIGHT, MoveType.NSIG, MoveType.NAIR, MoveType.RECOVERY]

    @staticmethod
    def is_down(move):
        return move in [MoveType.DLIGHT, MoveType.DSIG, MoveType.DAIR, MoveType.GROUNDPOUND]

    @staticmethod
    def is_side(move):
        return move in [MoveType.SLIGHT, MoveType.SSIG, MoveType.SAIR]

    @staticmethod
    def is_light(move):
        return move in [MoveType.NLIGHT, MoveType.DLIGHT, MoveType.SLIGHT, MoveType.NAIR, MoveType.DAIR, MoveType.SAIR]

    @staticmethod
    def is_heavy(move):
        return move in [MoveType.NSIG, MoveType.DSIG, MoveType.SSIG, MoveType.RECOVERY, MoveType.GROUNDPOUND]


class WeaponType(IntEnum):
    UNARMED = 0
    SPEAR = 1
    HAMMER = 2


class PlayerState(IntEnum):
    WalkingState = 0,
    StandingState = 1,
    TurnaroundState = 2,
    AirTurnaroundState = 3,
    SprintingState = 4,
    StunState = 5,
    InAirState = 6,
    DodgeState = 7,
    AttackState = 8,
    DashState = 9,
    BackDashState = 10,
    KOState = 11,
    TauntState = 12,

class MovementClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        # Three separate heads
        self.right_head = nn.Linear(128, 3)   # left / none / right
        self.jump_head = nn.Linear(128, 2)    # no jump / jump
        self.attack_head = nn.Linear(128, 2)  # no attack / attack

    def forward(self, x):
        feat = self.shared(x)
        right_logits = self.right_head(feat)
        jump_logits = self.jump_head(feat)
        attack_logits = self.attack_head(feat)
        return right_logits, jump_logits, attack_logits

class TTMovementClassifier(nn.Module):
    def __init__(self, state_dict, mesh_device):
        super().__init__()
        self.mesh_device = mesh_device

        # -----------------------------
        # Shared layers
        # -----------------------------
        self.fc1_w = ttnn.from_torch(
            state_dict["shared.0.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.fc1_b = ttnn.from_torch(
            state_dict["shared.0.bias"],
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.fc2_w = ttnn.from_torch(
            state_dict["shared.2.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.fc2_b = ttnn.from_torch(
            state_dict["shared.2.bias"],
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # -----------------------------
        # Output heads
        # -----------------------------
        self.right_w = ttnn.from_torch(
            state_dict["right_head.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.right_b = ttnn.from_torch(
            state_dict["right_head.bias"],
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.jump_w = ttnn.from_torch(
            state_dict["jump_head.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.jump_b = ttnn.from_torch(
            state_dict["jump_head.bias"],
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.attack_w = ttnn.from_torch(
            state_dict["attack_head.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.attack_b = ttnn.from_torch(
            state_dict["attack_head.bias"],
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs a forward pass of the multi-head TTNN policy.

        Args:
            obs (torch.Tensor): [batch_size, input_dim] input observations

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                right_logits [batch_size, 3]
                jump_logits [batch_size, 2]
                attack_logits [batch_size, 2]
        """

        # Convert input to bfloat16 and move to TTNN device
        obs = obs.to(torch.bfloat16)
        tt_obs = ttnn.from_torch(obs, device=self.mesh_device, layout=ttnn.TILE_LAYOUT)

        # Shared layers
        feat1 = ttnn.linear(tt_obs, self.fc1_w, bias=self.fc1_b, activation="relu")
        tt_obs.deallocate()  # free memory
        feat2 = ttnn.linear(feat1, self.fc2_w, bias=self.fc2_b, activation="relu")
        feat1.deallocate()  # free memory

        # Output heads
        right_tt = ttnn.linear(feat2, self.right_w, bias=self.right_b)
        jump_tt = ttnn.linear(feat2, self.jump_w, bias=self.jump_b)
        attack_tt = ttnn.linear(feat2, self.attack_w, bias=self.attack_b)
        feat2.deallocate()  # free shared features

        # Convert outputs back to PyTorch float32
        right_logits = ttnn.to_torch(right_tt).to(torch.float32)
        jump_logits = ttnn.to_torch(jump_tt).to(torch.float32)
        attack_logits = ttnn.to_torch(attack_tt).to(torch.float32)

        return right_logits, jump_logits, attack_logits

class SubmittedAgent(Agent):
    '''
    Input the **file_path** to your agent here for submission!
    '''

    def __init__(
            self,
            file_path: Optional[str] = None,
    ):
        super().__init__(file_path)
        self.opp_next_attack_data = None
        self.action = [0 for _ in range(10)]
        self.opp_last_state = None
        self.opp_attack_data = None
        self.opp_next_attack = -1
        self.time = 0
        self.reset()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1,1))
        

    def _initialize(self) -> None:
        #self.model = MovementClassifier().to(self.device)
        gdown.download("https://drive.google.com/file/d/1hanV_E2B6f59FOKstsaIeEUcwCuOvK0n/view?usp=sharing", "movement_classifier_c.pt", quiet=False)
        state_dict = torch.load("movement_classifier_c.pt", map_location=self.device)
        print(state_dict.keys())
        #self.model.load_state_dict(state_dict)
        #self.model.eval()

        # # Here you define your tttnn model and we extract the state dictionary of your custom neural network and pass it to your ttnn model
        self.model = TTMovementClassifier(state_dict, self.mesh_device)
        # # Once you have a ttnn model, we make the following models point to your ttnn model
        # # such that when you perform inference during a match, calling self.model(obs) will actually invoke the forward pass of our ttnn model
        # self.model.policy.features_extractor.model = self.tt_model
        # self.model.policy.vf_features_extractor.model = self.tt_model
        # self.model.policy.pi_features_extractor.model = self.tt_model

    frame_time = 1/30
    gravity = 17.808

    def reset(self) -> None:
        self.unpress_all_keys()
        self.time = 0
        self.opp_next_attack = -1
        self.opp_attack_data = None
        self.opp_last_state = None

    def move_left(self):
        self.action[1] = 1
        self.action[3] = 0

    def move_right(self):
        self.action[1] = 0
        self.action[3] = 1

    def stop_move_horizontal(self):
        self.action[1] = 0
        self.action[3] = 0

    def stop_move_vertical(self):
        self.action[0] = 0
        self.action[2] = 0

    def hold_up(self):
        self.action[0] = 1
        self.action[2] = 0

    def hold_down(self):
        self.action[0] = 0
        self.action[2] = 1

    def set_jump(self, jump=True):
        self.action[4] = 1 if jump else 0

    def dodge(self):
        self.stop_move_horizontal()
        self.stop_attacking()
        self.action[6] = 1

    def set_pickup_drop(self, pickup=True):
        self.action[5] = 1 if pickup else 0

    def set_light_attack(self, attack=True):
        if not attack: return
        self.action[7] = 1
        self.action[8] = 0

    def set_heavy_attack(self, attack=True):
        if not attack: return
        self.action[8] = 1
        self.action[7] = 0

    def press_string(self, keys):
        action_names = ["w", "a", "s", "d", "space", 'h', 'l', 'j', 'k', 'g']
        for key in keys:
            i = action_names.index(key)
            self.action[i] = 1

    def stop_attacking(self):
        self.action[7] = 0
        self.action[8] = 0

    def unpress_all_keys(self):
        self.action = [0 for _ in range(10)]

    @staticmethod
    def is_middle_gap(pos):
        return -2.2 <= pos[0] <= 2.2

    @staticmethod
    def is_off_right_side(pos):
        return pos[0] > 6

    @staticmethod
    def is_off_left_side(pos):
        return pos[0] < -6

    @staticmethod
    def is_towards_right_side(pos):
        return pos[0] > 2

    @staticmethod
    def is_towards_left_side(pos):
        return pos[0] < -2

    @staticmethod
    def below_lower_stage(pos):
        return pos[1] > 2.32

    @staticmethod
    def below_upper_stage(pos):
        return pos[1] > 0.4

    def move_jump_recover(self, jumps_left, in_air, pos, vel):
        if vel[1] < -1.8 or pos[1] < -3: return  # Still rising or too high
        self.stop_attacking()
        if not in_air or jumps_left != 0:
            self.set_jump(self.time % 2 == 0)  # Tap jump
        elif in_air:
            self.hold_up()
            self.set_heavy_attack()

    # Assume platform is centered at platPos and 1.8 units wide
    # Assume plrPos is centered at player and 0.928 unit wide
    @staticmethod
    def aligned_with_platform(plr_pos, plat_pos, width=0.9):
        return (plr_pos[0] + 0.4 >= plat_pos[0] - width) and (plr_pos[0] - 0.4 <= plat_pos[0] + width)

    @staticmethod
    def is_right_end_of_platform(plr_pos, plat_pos):
        return plr_pos[0] > plat_pos[0] + 0.3 and SubmittedAgent.aligned_with_platform(plr_pos, plat_pos)

    @staticmethod
    def is_left_end_of_platform(plr_pos, plat_pos):
        return plr_pos[0] < plat_pos[0] - 0.3 and SubmittedAgent.aligned_with_platform(plr_pos, plat_pos)

    @staticmethod
    def is_above_platform(plr_pos, plat_pos):
        return plr_pos[1] < plat_pos[1]

    @staticmethod
    def is_within_stage(plr_pos):
        return (2 < plr_pos[0] < 6) or (-6 < plr_pos[0] < -2)

    @staticmethod
    def is_above_something(plr_pos, plat_pos):
        return SubmittedAgent.is_within_stage(plr_pos) or (
                    plat_pos[0] + 0.5 > plr_pos[0] > plat_pos[0] - 0.5 and 0 < plat_pos[1] - plr_pos[1] < 2.5)

    def acceptable_end_state(self, plr_pos, plat_pos):
        return SubmittedAgent.is_within_stage(plr_pos) or \
            (plr_pos[1] < self.lerp(2.32, 0.4, (plr_pos[0] + 2) / 4)) or False
        # self.aligned_with_platform(plr_pos, plat_pos, 0.3)

    @staticmethod
    def distance(pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    @staticmethod
    def check_collision(plr_pos, facing_right, hitbox, opp_pos):
        m = 1 if facing_right else -1
        attack_box = [plr_pos[0] + hitbox[0] * m, plr_pos[1] + hitbox[1], hitbox[2], hitbox[3]]
        opp_box = [opp_pos[0], opp_pos[1], 0.928, 1.024]

        # AABB collision detection
        return attack_box[0] + attack_box[2] / 2 >= opp_box[0] - opp_box[2] / 2 and \
            attack_box[0] - attack_box[2] / 2 <= opp_box[0] + opp_box[2] / 2 and \
            attack_box[1] + attack_box[3] / 2 >= opp_box[1] - opp_box[3] / 2 and \
            attack_box[1] - attack_box[3] / 2 <= opp_box[1] + opp_box[3] / 2

    @staticmethod
    def solve_fall_pos(pos, vel, y_ground):
        a = 0.5 * SubmittedAgent.gravity
        b = vel[1]
        c = pos[1] - y_ground

        t_hit = None

        if abs(a) > 1e-12:
            disc = b * b - 4 * a * c
            if disc < 0:
                return None
            sqrt_d = math.sqrt(disc)
            t1 = (-b + sqrt_d) / (2 * a)
            t2 = (-b - sqrt_d) / (2 * a)
            candidates = [t for t in (t1, t2) if t >= 0]
            if not candidates:
                return None
            t_hit = min(candidates)
        else:
            if abs(b) < 1e-12:
                return None if abs(c) > 1e-12 else (pos[0], 0)
            t_hit = -c / b
            if t_hit < 0:
                return None

        x_hit = pos[0] + vel[0] * t_hit

        return x_hit, t_hit

    def predict(self, obs):
        self.time += 1
        self.unpress_all_keys()

        plr_pos = self.obs_helper.get_section(obs, 'player_pos')
        plr_vel = self.obs_helper.get_section(obs, 'player_vel')
        plr_in_air = round(self.obs_helper.get_section(obs, 'player_aerial')[0]) == 1
        plr_jumps = round(self.obs_helper.get_section(obs, 'player_jumps_left')[0])
        plr_weapon = round(self.obs_helper.get_section(obs, 'player_weapon_type')[0])
        plr_state = PlayerState(round(self.obs_helper.get_section(obs, 'player_state')[0]))
        plr_facing = round(self.obs_helper.get_section(obs, 'player_facing')[0]) == 1
        can_dodge = round(self.obs_helper.get_section(obs, 'player_dodge_timer')[0])

        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_vel = self.obs_helper.get_section(obs, 'opponent_vel')
        opp_state = PlayerState(round(self.obs_helper.get_section(obs, 'opponent_state')[0]))
        opp_move = round(self.obs_helper.get_section(obs, 'opponent_move_type')[0])
        opp_weapon = round(self.obs_helper.get_section(obs, 'opponent_weapon_type')[0])
        opp_stun = round(self.obs_helper.get_section(obs, 'opponent_stun_frames')[0])
        opp_in_air = round(self.obs_helper.get_section(obs, 'opponent_aerial')[0]) == 1
        opp_facing = round(self.obs_helper.get_section(obs, 'opponent_facing')[0]) == 1

        plat_pos = self.obs_helper.get_section(obs, 'player_moving_platform_pos')
        plat_vel = self.obs_helper.get_section(obs, 'player_moving_platform_vel')
        spawners = [self.obs_helper.get_section(obs, f'player_spawner_{i + 1}') for i in range(4)]

        if plr_weapon == 0:
            maintain_distance = 0.5
            if plr_pos[1] < opp_pos[1] - 1:
                maintain_distance = 0
        elif plr_weapon == 1:
            maintain_distance = 2
            if plr_pos[1] < opp_pos[1] - 2:
                maintain_distance = 1
        else:
            maintain_distance = 2
        target_x = None

        opp_safe = SubmittedAgent.acceptable_end_state(self, opp_pos, plat_pos)
        prio_spear = False

        if plr_weapon == WeaponType.UNARMED or (prio_spear and plr_weapon == WeaponType.HAMMER):
            closest = None
            for spawner in spawners:
                if not spawner[2] or (plr_weapon == WeaponType.HAMMER and spawner[2] == 3): continue
                dist_to_me = self.distance(spawner, plr_pos)
                if not closest or dist_to_me < closest:
                    target_x, closest = spawner[0], dist_to_me

            if closest and closest < 0.7:
                target_x = None
                if self.time % 2 == 1: self.set_pickup_drop()

        if not target_x and opp_state != PlayerState.KOState and opp_safe:
            eff_pos = opp_pos
            if opp_pos[1] < plr_pos[1]:
                eff_pos[0], _ = self.solve_fall_pos(opp_pos, opp_vel, plr_pos[1])
                if eff_pos[0] is None:
                    eff_pos[0] = opp_pos[0]

            if self.is_towards_right_side(eff_pos):
                factor = -1 if eff_pos[0] > 4 else 1
                target_x = eff_pos[0] + factor * maintain_distance
            elif self.is_towards_left_side(eff_pos):
                factor = -1 if eff_pos[0] > -4 else 1
                target_x = eff_pos[0] + factor * maintain_distance
            else:
                target_x = eff_pos[0] + (-maintain_distance if plr_pos[0] < eff_pos[0] else maintain_distance)

        move_prio = self.do_movement(target_x, plr_pos, opp_pos, plr_vel, opp_vel, plr_in_air, plr_jumps, plat_pos, plat_vel)

        if move_prio < MovePrio.ATTACK_OVERRIDE:
            self.try_attack(plr_weapon, plr_in_air, plr_pos, plr_facing, opp_in_air, opp_pos, plr_vel, opp_vel, plat_pos, opp_stun,
                   allow_risky=True)

        if opp_state == PlayerState.AttackState and self.opp_last_state != PlayerState.AttackState:
            try:
                move_data = self.attack_data()[opp_weapon][opp_in_air][opp_move]
            except KeyError:
                move_data = self.attack_data()[opp_weapon][not opp_in_air][opp_move]
            delay = move_data["delay"] - 1
            if can_dodge <= delay:
                self.opp_next_attack = self.time + delay
                self.opp_next_attack_data = {
                    "move": move_data,
                    "pos": opp_pos,
                    "facing": opp_facing
                }
                self.opp_attack_data = MoveType(opp_move) in MoveType.prio_dodge()
                #
        if self.opp_attack_data and self.opp_next_attack != -1:
            self.stop_attacking()
        if self.time == self.opp_next_attack:
            if self.distance(plr_pos, opp_pos) < 2.5:
                m_hit = self.opp_next_attack_data["move"]["hitbox"]
                if m_hit is not None:
                    if self.check_collision(opp_pos, opp_facing, m_hit, plr_pos):
                        if can_dodge == 0:
                            self.dodge()
                else:
                    if can_dodge == 0:
                        self.dodge()
            self.opp_next_attack = -1

        self.opp_last_state = opp_state
        return self.action

    def do_movement(self, target_x, plr_pos, opp_pos, plr_vel, opp_vel, plr_in_air, plr_jumps, plat_pos, plat_vel):
        if target_x:
            target_x = max(-6, min(6, target_x))
        else:
            target_x = plr_pos[0]
        distance = self.distance(plr_pos, opp_pos)
        next_attack = 1 if self.opp_next_attack != -1 else 0
        counter_stats = None if next_attack == 0 or ("counter" not in self.opp_next_attack_data) else self.opp_next_attack_data["counter"]
        jump_attack = not counter_stats or "do_not_jump" not in counter_stats
        jump_attack = 1 if jump_attack else 0
        plr_in_air = 1 if plr_in_air else 0
        plr_vel = [1 if plr_vel[0] > 0 else -1, plr_vel[1]]
        plat_vel = [1 if plat_vel[0] > 0 else -1, plat_vel[1]]

        pos_scale = 18

        obs = [target_x / pos_scale,
                plr_pos[0] / pos_scale, plr_pos[1] / pos_scale,
                opp_pos[0] / pos_scale, opp_pos[1] / pos_scale,
                plr_vel[0], plr_vel[1] / 10,
                plr_in_air, plr_jumps/3,
                plat_pos[0] / pos_scale, plat_pos[1] / pos_scale,
                plat_vel[0], plat_vel[1] / 10,
                distance / pos_scale, next_attack, jump_attack
               ]

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
            r_logits, j_logits, a_logits = self.model(obs_t)
            right = torch.argmax(r_logits, dim=0).item()
            jump = torch.argmax(j_logits, dim=0).item()
            attack = torch.argmax(a_logits, dim=0).item()

        right = [-1, 0, 1][right]
        jump = round(jump)
        attack = round(attack)

        if right == 1:
            self.move_right()
        elif right == -1:
            self.move_left()
        if jump == 1:
            self.move_jump_recover(plr_jumps, plr_in_air, plr_pos, plr_vel)
        return MovePrio.ATTACK_FIRST if attack == 1 else MovePrio.ATTACK_OVERRIDE

    def try_attack(self, plr_weapon, plr_in_air, plr_pos, plr_facing, opp_in_air, opp_pos, plr_vel, opp_vel, plat_pos, stun,
                   allow_risky=True):
        data = self.attack_data()
        attacks = data[plr_weapon][plr_in_air]

        valid = []
        for move, move_data in attacks.items():
            delay = move_data["delay"]
            duration = move_data["safety"]["duration"]
            hitbox = move_data["hitbox"]

            if not self.check_valid_attack(move_data, hitbox, plr_pos, plr_vel, plr_facing, delay, plr_in_air,
                                           allow_risky, plat_pos):
                continue
            opp_pos_future = self.future_pos(opp_pos, opp_vel, opp_in_air, delay, stun, wall=plr_pos[0])
            plr_pos_future = self.future_pos(plr_pos, plr_vel, opp_in_air, delay, delay)
            if not self.check_attack_hit(plr_pos, plr_facing, opp_in_air, opp_pos, plr_vel, opp_vel, delay, duration,
                                         hitbox, stun):
                continue
            dist = self.distance(opp_pos, plr_pos)
            prio = move_data["safety"]["prio"]
            weight = 0
            if prio == "far":
                weight += 1 if dist > 2.5 else -1

            valid.append((weight, move, opp_pos_future, plr_pos_future))

        valid.sort(key=lambda x: x[0], reverse=True)

        if len(valid) > 0:
            weight, move, opp_pos_future, plr_pos_future = valid[0]
            if MoveType.is_up(move):
                self.hold_up()
                self.stop_move_horizontal()
            elif MoveType.is_down(move):
                self.hold_down()
                self.stop_move_horizontal()
            elif MoveType.is_side(move):
                self.move_right() if opp_pos[0] > plr_pos[0] else self.move_left()
                self.stop_move_vertical()
            if MoveType.is_light(move):
                self.set_light_attack()
            elif MoveType.is_heavy(move):
                self.set_heavy_attack(self.time % 2 == 0)
            return True

        for move, move_data in attacks.items():
            delay = move_data["delay"]
            hitbox = move_data["hitbox"]
            duration = move_data["safety"]["duration"]
            if not self.check_valid_attack(move_data, hitbox, plr_pos, plr_vel, not plr_facing, delay, plr_in_air,
                                           allow_risky, plat_pos):
                continue

            if not self.check_attack_hit(plr_pos, not plr_facing, opp_in_air, opp_pos, plr_vel, opp_vel, delay, duration, hitbox, stun):
                continue
            self.move_left() if plr_facing else self.move_right()
            return True
        return False

    def check_attack_hit(self, plr_pos, plr_facing, opp_in_air, opp_pos, plr_vel, opp_vel, delay, duration, hitbox, stun, samples = 4):
        d = (duration - delay)/float(samples)
        for i in range(samples):
            t = int(delay + i*d)
            opp_pos_future = self.future_pos(opp_pos, opp_vel, opp_in_air, t, stun - 1, wall=plr_pos[0])
            plr_pos_future = self.future_pos(plr_pos, plr_vel, opp_in_air, t, t)
            if self.check_collision(plr_pos_future, plr_facing, hitbox, opp_pos_future):
                return True
        return False

    def check_valid_attack(self, move_data, hitbox, plr_pos, plr_vel, plr_facing, delay, plr_in_air, allow_risky, plat_pos):
        if hitbox is None: return False
        if self.time + delay > self.opp_next_attack != -1:
            return False
        if "safety" in move_data:
            safety_stats = move_data["safety"]
            if not allow_risky and ("risky" in safety_stats):
                return False
            duration = safety_stats["duration"]
            if "ignore_future" in move_data:
                return True

            end_pos = self.predict_pos(plr_pos, plr_vel, duration, plr_in_air)
            #end_pos = self.position_at_end(plr_pos, plr_vel, [-10,-10], duration)
            if not plr_in_air and not self.is_above_something(end_pos, plat_pos):
                return False
            safe_after = self.acceptable_end_state(end_pos, plat_pos)
            if "offset" in move_data:
                end_pos = self.predict_pos(plr_pos + move_data["offset"] * (1 if plr_facing else -1), plr_vel, duration, plr_in_air)
                safe_after = safe_after and self.acceptable_end_state(end_pos, plat_pos)
                if not plr_in_air and not self.is_above_something(end_pos, plat_pos):
                    return False
            if not safe_after:
                return False
        return True

    def predict_pos(self, pos, vel, time, in_air=True):
        x = pos[0] + time * self.frame_time * vel[0]
        y = pos[1] + self.predict_y_disp(vel, time) * (1 if in_air else 0)
        return self.limit_to_platforms([x,y])

    @staticmethod
    def limit_to_platforms(pos):
        if -6 < pos[0] < -2:
            pos[1] = max(2.32, pos[1])
        if 2 < pos[0] < 6:
            pos[1] = max(0.4, pos[1])
        return pos
    def predict_y_disp(self, vel, time):
        return vel[1] * time * self.frame_time + 0.5 * self.gravity * (time - 1) * time * (self.frame_time ** 2)
        return time * self.frame_time * vel[1] + 1 / 2 * self.gravity * (time * self.frame_time * vel[1]) * (time *
                    self.frame_time * vel[1])

    def future_pos(self, pos, vel, in_air, delay, stun, wall = None):
        y_prediction = self.predict_y_disp(vel, delay)
        mag_x = self.lerp(0.7, 1, stun / delay) * (0.7 if in_air else 0.3)
        mag_y = self.lerp(1, 1, stun / delay) * (1 if in_air else 0)

        pred = [
            pos[0] + mag_x * vel[0] * self.frame_time * delay,
            pos[1] + mag_y * y_prediction
        ]
        if wall:
            if pos[0] > wall > pred[0]:
                pred[0] = wall + 0.1
            if pos[0] < wall < pred[0]:
                pred[0] = wall - 0.1
        return self.limit_to_platforms(pred)

    @staticmethod
    def lerp(a, b, t):
        return a + (b - a) * min(max(t, 0), 1)

    @staticmethod
    def attack_data():
        return {
            WeaponType.UNARMED: {
                True: {
                    MoveType.NAIR: {
                        "hitbox":(0.6, 0, 0.5, 0.8),
                        "delay":4,
                        "safety": {
                            "prio":"default",
                            "duration": 16,
                        },
                    },
                    MoveType.DAIR: {
                        "hitbox":(1, 1, 0.5, 0.5),
                        "delay":6,
                        "safety": {
                            "prio": "default",
                            "duration": 25,
                            "offset": 1.5
                        }
                    },
                    MoveType.SAIR: {
                        "hitbox": None,
                        "delay": 5,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                    },
                    MoveType.GROUNDPOUND: {
                        "hitbox":(0, 2, 0.1, 4),
                        "delay":10.,
                        "safety": {
                            "prio": "far",
                            "duration": 25,
                            "risky":True,
                            "cancels_x": True
                        },
                    },
                    MoveType.RECOVERY: {
                        "hitbox": None,
                        "delay": 6,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                    },
                },
                False: {
                    MoveType.NLIGHT: {
                        "hitbox": (0.5, 0, 0.5, 0.8),
                        "delay": 3,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                    },
                    MoveType.DLIGHT: {
                        "hitbox": None,
                        "delay": 5,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                    },
                    MoveType.SLIGHT: {
                        "hitbox": None,
                        "delay": 5,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                    },
                    MoveType.NSIG: {
                        "hitbox": None,
                        "delay": 10,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                    },
                    MoveType.DSIG: {
                        "hitbox": (0.2, 0.25, 1.7, 0.5),
                        "delay": 11,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                    },
                    MoveType.SSIG: {
                        "hitbox": None,
                        "delay": 11,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                    },
                }
            },
            WeaponType.SPEAR: {
                True: {
                    MoveType.NAIR: {
                        "hitbox": (0.3, 0, 1.7, 1.8),
                        "delay": 8,
                        "safety": {
                            "prio": "default",
                            "duration": 22,
                        },
                        "counter": {
                            "do_not_jump": True
                        },
                    },
                    MoveType.DAIR: {
                        "hitbox": (0.2, 0.5, 0.3, 1),
                        "delay": 6,
                        "safety": {
                            "prio": "default",
                            "duration": 15,
                        },
                    },
                    MoveType.SAIR: {
                        "hitbox": (1.6, 0.15, 0.4, 0.8),
                        "delay": 7,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                    },
                    MoveType.GROUNDPOUND: {
                        "hitbox": (0, 1.5, 2, 3.5),
                        "delay": 9,
                        "safety": {
                            "prio": "far",
                            "risky": True,
                            "duration": 36,
                            "cancels_x": True
                        },
                    },
                    MoveType.RECOVERY: {
                        "hitbox": (0, 1, 1, 2),
                        "delay": 9,
                        "safety": {
                            "prio": "default",
                            "ignore_future": True,
                            "duration": 22,
                        },
                    },
                },
                False: {
                    MoveType.DSIG: {
                        "hitbox": (0.7, 0, 1.5, 2),
                        "delay": 16.,
                        "safety": {
                            "prio": "far",
                            "duration": 25,
                            "offset": -2,
                        },
                        "counter": {
                            "do_not_jump": True
                        },
                    },
                    MoveType.NSIG: {
                        "hitbox": (0.6, -0.8, 2, 0.7),
                        "delay": 11,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                        "counter": {
                            "do_not_jump": True
                        },
                    },
                    MoveType.NLIGHT: {
                        "hitbox": (1.3, 0, 0.7, 0.3),
                        "delay": 4,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                        "counter": {
                            "do_not_jump": True
                        },
                    },
                    MoveType.DLIGHT: {
                        "hitbox": None,
                        "delay": 5,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                    },
                    MoveType.SLIGHT: {
                        "hitbox": (1, 0, 2, 1.5),
                        "delay": 7,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                    },
                    MoveType.SSIG: {
                        "hitbox": None,
                        "delay": 16,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                    },
                }
            },
            WeaponType.HAMMER: {
                True: {
                    MoveType.NAIR: {
                        "hitbox": (0.4, -1.3, 0.9, 0.9),
                        "delay": 9,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                    },
                    MoveType.DAIR: {
                        "hitbox": None,
                        "delay": 6,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                    },
                    MoveType.SAIR: {
                        "hitbox": (1.1, 0, 0.7, 0.7),
                        "delay": 7.,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                    },
                    MoveType.GROUNDPOUND: {
                        "hitbox": (0, 1, 0.5, .7),
                        "delay": 13,
                        "safety": {
                            "prio": "default",
                            "duration": 35,
                            "risky": True,
                            "cancels_x": True
                        }
                    },
                    MoveType.RECOVERY: {
                        "hitbox": (-0.3, 0.5, 0.8, 1),
                        "delay": 7.,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                        },
                    },
                },
                False: {
                    MoveType.NSIG: {
                        "hitbox": (0.8, -0.6, 1, 1.5),
                        "delay": 16.,
                        "safety": {
                            "prio": "default",
                            "duration": 24,
                        },
                        "counter": {
                            "do_not_jump": True
                        }
                    },
                    MoveType.NLIGHT: {
                        "hitbox": (1, -0.6, 1.5, 0.5),
                        "delay": 4.,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                            "offset": 0.7,
                        },
                    },
                    MoveType.SSIG: {
                        "hitbox": None, #(1.3, -0.4, 2, 1.2),
                        "delay": 17.,
                        "safety": {
                            "prio": "far",
                            "duration": 30,
                            "offset": 1.5,
                            "risky": True,
                        },
                    },
                    MoveType.DLIGHT: {
                        "hitbox": (0.7, 0.5, 1.6, 0.3),
                        "delay": 5.,
                        "safety": {
                            "prio": "default",
                            "duration": 12,
                        },
                    },
                    MoveType.SLIGHT: {
                        "hitbox": (2, -0.25, 1.5, 1.25),
                        "delay": 7.,
                        "safety": {
                            "prio": "default",
                            "duration": 16,
                            "offset": 1,
                        },
                    },
                    MoveType.DSIG: {
                        "hitbox": None,
                        "delay": 16.,
                        "safety": {
                            "prio": "default",
                            "duration": 24,
                        },
                    },
                }
            }
        }