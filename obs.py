
import copy
import math
import random

import numpy as np
from typing import Any, List

from gym import Space
from gym.spaces import Tuple, Box
from rlgym.gym import Gym
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.common_values import BOOST_LOCATIONS
import torch

from numba import njit
import scipy


# inspiration from Raptor (Impossibum) and Necto (Rolv/Soren)


class CoyoteObsBuilder(ObsBuilder):
    def __init__(self, tick_skip=8, team_size=1, expanding: bool = True, extra_boost_info: bool = True,
                 embed_players=False, stack_size=0, action_parser=None, env: Gym = None, infinite_boost_odds=0,
                 only_closest_opp=False,
                 selector=False,
                 end_object_choice=None,
                 remove_other_cars=False,
                 zero_other_cars=False,
                 override_cars=False,
                 # obs_output=None,
                 obs_info=None,
                 add_handbrake=False,
                 add_jumptime=False,
                 add_fliptime=False,
                 add_airtime=False,
                 add_boosttime=False,
                 dodge_deadzone=0.8,
                 flip_dir=True,
                 end_object: PhysicsObject = None,
                 ):
        super().__init__()
        self.flip_dir = flip_dir
        self.end_object = end_object
        assert add_boosttime == add_airtime == add_fliptime == add_jumptime == add_handbrake, "All timers must match"
        self.obs_info = obs_info
        # self.obs_output = obs_output
        self.override_cars = override_cars
        self.zero_other_cars = zero_other_cars
        self.remove_other_cars = remove_other_cars
        self.expanding = expanding
        self.only_closest_opp = only_closest_opp
        self.extra_boost_info = extra_boost_info
        self.POS_STD = 2300
        self.VEL_STD = 2300
        self.ANG_STD = 5.5
        self.BOOST_TIMER_STD = 10
        self.DEMO_TIMER_STD = 3
        self.dummy_player = [0] * 35
        self.dummy_tm8 = [0] * 35
        self.dummy_tm8[33] = 1
        self.boost_locations = np.array(BOOST_LOCATIONS)
        self.inverted_boost_locations = self.boost_locations[::-1]
        self.boost_timers = np.zeros(self.boost_locations.shape[0])
        self.inverted_boost_timers = np.zeros(self.boost_locations.shape[0])
        self.boosts_availability = np.zeros(self.boost_locations.shape[0])
        self.inverted_boosts_availability = np.zeros(
            self.boost_locations.shape[0])
        self.boost_values = np.ones(self.boost_locations.shape[0]) * 0.12
        np.put(self.boost_values, [3, 4, 15, 18, 29, 30], 1)
        self.boost_objs = []
        self.inverted_boost_objs = []
        self.state = None
        self.time_interval = tick_skip / 120
        self.demo_timers = None
        self.num_players = team_size * 2
        self.generic_obs = None
        self.blue_obs = None
        self.orange_obs = None
        self.embed_players = embed_players
        self.selector = selector
        self.default_action = np.zeros(8)
        self.stack_size = stack_size
        self.action_stacks = {}
        self.model_action_stacks = {}
        self.action_size = self.default_action.shape[0]
        self.action_parser = action_parser
        if self.action_parser is not None:
            self.model_action_size = action_parser.get_model_action_size()
        self.env = env
        self.infinite_boost_odds = infinite_boost_odds
        self.infinite_boost_episode = False
        self.end_object_choice = end_object_choice
        self.end_object_tracker = 0
        if end_object_choice is not None and end_object_choice == "random":
            self.end_object_tracker = 0
        elif end_object_choice is not None:
            self.end_object_tracker = int(self.end_object_choice)
        self.big_boosts = [BOOST_LOCATIONS[i] for i in [3, 4, 15, 18, 29, 30]]
        self.big_boosts = np.asarray(self.big_boosts)
        self.big_boosts[:, -1] = 18
        self.add_boosttime = add_boosttime
        self.add_jumptime = add_jumptime
        self.add_fliptime = add_fliptime
        self.add_airtime = add_airtime
        self.add_handbrake = add_handbrake
        self.dodge_deadzone = dodge_deadzone
        self.any_timers = add_boosttime or add_jumptime or add_fliptime or add_airtime or add_handbrake

        self.boosttimes = [0] * 6
        self.jumptimes = [0] * 6
        self.fliptimes = [0] * 6
        self.has_flippeds = [False] * 6
        self.has_doublejumpeds = [False] * 6
        self.flipdirs = [[0] * 2 for _ in range(6)]
        self.airtimes = [0] * 6
        self.on_grounds = [False] * 6
        self.prev_prev_actions = [[0] * 8 for _ in range(6)]
        self.is_jumpings = [False] * 6
        self.has_jumpeds = [False] * 6
        self.handbrakes = [0] * 6

    def reset(self, initial_state: GameState):
        self.state = None
        if self.obs_info is None:
            self.boost_timers = np.zeros(self.boost_locations.shape[0])
            self.inverted_boost_timers = np.zeros(
                self.boost_locations.shape[0])
            self.demo_timers = np.zeros(
                max(p.car_id for p in initial_state.players) + 1)
            self.blue_obs = []
            self.orange_obs = []

        self.action_stacks = {}
        if self.stack_size != 0 and not self.selector:
            for p in initial_state.players:
                self.action_stacks[p.car_id] = np.concatenate(
                    [self.default_action] * self.stack_size)

        self.model_action_stacks = {}
        if self.stack_size != 0 and self.selector:
            for p in initial_state.players:
                self.model_action_stacks[p.car_id] = [0.] * self.stack_size

        if self.action_parser is not None:
            self.action_parser.reset(initial_state)

        if self.env is not None:
            if random.random() <= self.infinite_boost_odds:
                self.env.update_settings(boost_consumption=0)
                self.infinite_boost_episode = True
            else:
                self.env.update_settings(boost_consumption=1)
                self.infinite_boost_episode = False

        if self.add_boosttime:
            self.boosttimes = np.zeros(
                max(p.car_id for p in initial_state.players) + 1)

        if self.add_jumptime:
            self.jumptimes = np.zeros(
                max(p.car_id for p in initial_state.players) + 1)

        if self.add_fliptime:
            self.fliptimes = np.zeros(
                max(p.car_id for p in initial_state.players) + 1)
            self.has_flippeds = [False] * (max(p.car_id for p in initial_state.players) + 1)
            self.has_doublejumpeds = [False] * (max(p.car_id for p in initial_state.players) + 1)
            self.flipdirs = [[0] * 2 for _ in range(max(p.car_id for p in initial_state.players) + 1)]

        if self.add_airtime:
            self.airtimes = np.zeros(
                max(p.car_id for p in initial_state.players) + 1)

        if self.add_jumptime or self.add_fliptime or self.add_airtime:
            self.prev_prev_actions = [[0] * 8 for _ in range(max(p.car_id for p in initial_state.players) + 1)]
            self.is_jumpings = [False] * (max(p.car_id for p in initial_state.players) + 1)
            self.has_jumpeds = [False] * (max(p.car_id for p in initial_state.players) + 1)
            self.on_grounds = [False] * (max(p.car_id for p in initial_state.players) + 1)
            for p in initial_state.players:
                self.on_grounds[p.car_id] = p.on_ground

        if self.add_handbrake:
            self.handbrakes = np.zeros(
                max(p.car_id for p in initial_state.players) + 1)

    def pre_step(self, state: GameState):
        self.state = state
        # create player/team agnostic items (do these even exist?)
        self._update_timers(state)
        # create team specific things
        self.blue_obs = self.boost_timers / self.BOOST_TIMER_STD
        self.orange_obs = self.inverted_boost_timers / self.BOOST_TIMER_STD

        if self.env is not None:
            if self.infinite_boost_episode:
                for player in state.players:
                    player.boost_amount = 2
            else:
                for player in state.players:
                    player.boost_amount /= 1

        if self.end_object is not None and \
                not (self.end_object.position[0] == self.end_object.position[1] == self.end_object.position[2] == -1):
            state.ball.position[0] = self.end_object.position[0]
            state.ball.position[1] = self.end_object.position[1]
            state.ball.position[2] = self.end_object.position[2]
            state.ball.linear_velocity = np.asarray([0, 0, 0])
            state.ball.angular_velocity = np.asarray([0, 0, 0])

    def _update_timers(self, state: GameState):
        current_boosts = state.boost_pads
        boost_locs = self.boost_locations
        demo_states = [[p.car_id, p.is_demoed] for p in state.players]

        for i in range(len(current_boosts)):
            if current_boosts[i] == self.boosts_availability[i]:
                if self.boosts_availability[i] == 0:
                    self.boost_timers[i] = max(
                        0, self.boost_timers[i] - self.time_interval)
            else:
                if self.boosts_availability[i] == 0:
                    self.boosts_availability[i] = 1
                    self.boost_timers[i] = 0
                else:
                    self.boosts_availability[i] = 0
                    if boost_locs[i][2] == 73:
                        self.boost_timers[i] = 10.0
                    else:
                        self.boost_timers[i] = 4.0
        self.boosts_availability = current_boosts
        self.inverted_boost_timers = self.boost_timers[::-1]
        self.inverted_boosts_availability = self.boosts_availability[::-1]

        for cid, dm in demo_states:
            if dm == True:  # Demoed
                prev_timer = self.demo_timers[cid]
                if prev_timer > 0:
                    self.demo_timers[cid] = max(
                        0, prev_timer - self.time_interval)
                else:
                    self.demo_timers[cid] = 3
            else:  # Not demoed
                self.demo_timers[cid] = 0

    def _update_addl_timers(self, player: PlayerData, state: GameState, prev_actions: np.ndarray):
        cid = player.car_id

        # if this player was not boosting last tick and their boosttime timer means they actually stopped boosting, set to 0
        if prev_actions[6] == 0 and self.boosttimes[cid] == 12:
            self.boosttimes[cid] = 0
        # otherwise, just increment the boosttime
        else:
            self.boosttimes[cid] += self.time_interval * 120
            self.boosttimes[cid] = min(12, self.boosttimes[cid])

        # update jumptime
        if self.on_grounds[cid] and not self.is_jumpings[cid]:
            self.has_jumpeds[cid] = False

        if self.is_jumpings[cid]:
            # JUMP_MIN_TIME = 3 ticks
            # JUMP_MAX_TIME = 24 ticks
            if not ((self.jumptimes[cid] < 3 or prev_actions[5] == 1) and self.jumptimes[cid] < 24):
                self.is_jumpings[cid] = self.jumptimes[cid] < 3
        elif prev_actions[5] == 1 and self.prev_prev_actions[cid][5] == 0 and self.on_grounds[cid]:
            self.is_jumpings[cid] = True
            self.jumptimes[cid] = 0

        if self.is_jumpings[cid]:
            self.has_jumpeds[cid] = True
            self.jumptimes[cid] += self.time_interval * 120
            self.jumptimes[cid] = min(
                24, self.jumptimes[cid])
        else:
            self.jumptimes[cid] = 0

        # update airtime and fliptime
        if player.on_ground:
            self.has_doublejumpeds[cid] = False
            self.has_flippeds[cid] = False
            self.airtimes[cid] = 0
            self.fliptimes[cid] = 0
            self.flipdirs[cid] = [0, 0]
            self.on_grounds[cid] = True
        else:
            if self.has_jumpeds[cid] and not self.is_jumpings[cid]:
                self.airtimes[cid] += self.time_interval * 120
                # DOUBLEJUMP_MAX_DELAY = 150 ticks
                self.airtimes[cid] = min(
                    150, self.airtimes[cid])
            else:
                self.airtimes[cid] = 0
            if self.has_jumpeds[cid] and (prev_actions[5] == 1 and self.prev_prev_actions[cid][5] == 0) and \
                    self.airtimes[cid] < 150:
                if not self.has_doublejumpeds[cid] and not self.has_flippeds[cid]:
                    should_flip = max(max(abs(prev_actions[3]), abs(prev_actions[2])), abs(
                        prev_actions[4])) >= self.dodge_deadzone
                    if should_flip:
                        self.fliptimes[cid] = 0
                        self.has_flippeds[cid] = True
                        flipdir = np.asarray(
                            [-prev_actions[2], prev_actions[3] + prev_actions[4]])
                        if np.any(flipdir):
                            self.flipdirs[cid] = list(
                                flipdir / np.linalg.norm(flipdir))
                        else:
                            self.flipdirs[cid] = [0,0]
                    else:
                        self.has_doublejumpeds[cid] = True
        if self.has_flippeds[cid]:
            self.fliptimes[cid] += self.time_interval * 120
            # FLIP_TORQUE_TIME = 78 ticks
            self.fliptimes[cid] = min(
                78, self.fliptimes[cid])

        # update handbrake
        if prev_actions[7] == 1:
            # POWERSLIDE_RISE_RATE = 5
            self.handbrakes[cid] += 5 * self.time_interval
            self.handbrakes[cid] = min(
                1, self.handbrakes[cid])
        else:
            # POWERSLIDE_FALL_RATE = 2
            self.handbrakes[cid] -= 2 * self.time_interval
            self.handbrakes[cid] = max(
                0, self.handbrakes[cid])

    def create_ball_packet(self, ball: PhysicsObject):
        p = [
            ball.position[0] / self.POS_STD, ball.position[1] /
            self.POS_STD, ball.position[2] / self.POS_STD,
            ball.linear_velocity[0] /
            self.VEL_STD, ball.linear_velocity[1] / self.VEL_STD,
            ball.linear_velocity[2] / self.VEL_STD,
            ball.angular_velocity[0] /
            self.ANG_STD, ball.angular_velocity[1] / self.ANG_STD,
            ball.angular_velocity[2] / self.ANG_STD,
            math.sqrt(
                ball.linear_velocity[0] ** 2 + ball.linear_velocity[1] ** 2 + ball.linear_velocity[2] ** 2) / 2300,
            int(ball.position[2] <= 100),
        ]
        return p

    @staticmethod
    @njit(cache=True)
    def create_player_packet_njit(car_position: np.ndarray,
                                  car_linear_velocity: np.ndarray,
                                  car_angular_velocity: np.ndarray,
                                  fwd: np.ndarray,
                                  up: np.ndarray,
                                  boost: float,
                                  on_ground: bool,
                                  has_jump: bool,
                                  has_flip: bool,
                                  is_demoed: bool,
                                  demo_timer: float,
                                  pos_std: int,
                                  vel_std: int,
                                  ang_std: float,
                                  ball_position: np.ndarray,
                                  ball_linear_velocity: np.ndarray,
                                  prev_act: np.ndarray,
                                  ):
        pos_diff = ball_position - car_position
        vel_diff = ball_linear_velocity - car_linear_velocity
        return [
            car_position[0] / pos_std, car_position[1] /
            pos_std, car_position[2] / pos_std,
            car_linear_velocity[0] / vel_std, car_linear_velocity[1] / vel_std,
            car_linear_velocity[2] / vel_std,
            car_angular_velocity[0] /
            ang_std, car_angular_velocity[1] / ang_std,
            car_angular_velocity[2] / ang_std,
            pos_diff[0] / pos_std, pos_diff[1] /
            pos_std, pos_diff[2] / pos_std,
            vel_diff[0] / vel_std, vel_diff[1] /
            vel_std, vel_diff[2] / vel_std,
            fwd[0], fwd[1], fwd[2],
            up[0], up[1], up[2],
            np.sqrt(car_linear_velocity[0] ** 2 + car_linear_velocity[1]
                    ** 2 + car_linear_velocity[2] ** 2) / 2300,
            boost,
            int(on_ground),
            int(has_flip),
            int(is_demoed),
            int(has_jump),
            demo_timer,
            prev_act[0], prev_act[1], prev_act[2], prev_act[3], prev_act[4], prev_act[5], prev_act[6], prev_act[7],
        ]

    @staticmethod
    @njit(cache=True)
    def create_player_packet_timers_njit(car_position: np.ndarray,
                                         car_linear_velocity: np.ndarray,
                                         car_angular_velocity: np.ndarray,
                                         fwd: np.ndarray,
                                         up: np.ndarray,
                                         boost: float,
                                         on_ground: bool,
                                         has_jump: bool,
                                         has_flip: bool,
                                         is_demoed: bool,
                                         demo_timer: float,
                                         pos_std: int,
                                         vel_std: int,
                                         ang_std: float,
                                         ball_position: np.ndarray,
                                         ball_linear_velocity: np.ndarray,
                                         prev_act: np.ndarray,
                                         boosttime: float,
                                         jumptime: float,
                                         airtime: float,
                                         fliptime: float,
                                         braketime: float,
                                         flip_dir_1: float,
                                         flip_dir_2: float,
                                         ):
        pos_diff = ball_position - car_position
        vel_diff = ball_linear_velocity - car_linear_velocity
        return [
            car_position[0] / pos_std, car_position[1] /
            pos_std, car_position[2] / pos_std,
            car_linear_velocity[0] / vel_std, car_linear_velocity[1] / vel_std,
            car_linear_velocity[2] / vel_std,
            car_angular_velocity[0] /
            ang_std, car_angular_velocity[1] / ang_std,
            car_angular_velocity[2] / ang_std,
            pos_diff[0] / pos_std, pos_diff[1] /
            pos_std, pos_diff[2] / pos_std,
            vel_diff[0] / vel_std, vel_diff[1] /
            vel_std, vel_diff[2] / vel_std,
            fwd[0], fwd[1], fwd[2],
            up[0], up[1], up[2],
            np.sqrt(car_linear_velocity[0] ** 2 + car_linear_velocity[1]
                    ** 2 + car_linear_velocity[2] ** 2) / 2300,
            boost,
            int(on_ground),
            int(has_flip),
            int(is_demoed),
            int(has_jump),
            demo_timer,
            prev_act[0], prev_act[1], prev_act[2], prev_act[3], prev_act[4], prev_act[5], prev_act[6], prev_act[7],
            boosttime / 12,
            jumptime / 24,
            airtime / 150,
            fliptime / 78,
            braketime,
            flip_dir_1,
            flip_dir_2
        ]

    def create_player_packet(self, player: PlayerData, car: PhysicsObject, ball: PhysicsObject, prev_act: np.ndarray,
                             prev_model_act: np.ndarray):
        pos_diff = ball.position - car.position
        vel_diff = ball.linear_velocity - car.linear_velocity
        fwd = car.forward()
        up = car.up()
        p = [
            car.position[0] / self.POS_STD, car.position[1] /
            self.POS_STD, car.position[2] / self.POS_STD,
            car.linear_velocity[0] /
            self.VEL_STD, car.linear_velocity[1] / self.VEL_STD,
            car.linear_velocity[2] / self.VEL_STD,
            car.angular_velocity[0] /
            self.ANG_STD, car.angular_velocity[1] / self.ANG_STD,
            car.angular_velocity[2] / self.ANG_STD,
            pos_diff[0] / self.POS_STD, pos_diff[1] /
            self.POS_STD, pos_diff[2] / self.POS_STD,
            vel_diff[0] / self.VEL_STD, vel_diff[1] /
            self.VEL_STD, vel_diff[2] / self.VEL_STD,
            fwd[0], fwd[1], fwd[2],
            up[0], up[1], up[2],
            math.sqrt(car.linear_velocity[0] ** 2 + car.linear_velocity[1]
                      ** 2 + car.linear_velocity[2] ** 2) / 2300,
            player.boost_amount,
            int(player.on_ground),
            int(player.has_flip),
            int(player.is_demoed),
            int(player.has_jump),
            self.demo_timers[player.car_id] / self.DEMO_TIMER_STD,
            prev_act[0], prev_act[1], prev_act[2], prev_act[3], prev_act[4], prev_act[5], prev_act[6], prev_act[7],
        ]

        if self.stack_size != 0:
            if self.selector:
                self.model_add_action_to_stack(prev_model_act, player.car_id)
                p.extend(list(self.model_action_stacks[player.car_id]))

            else:
                self.add_action_to_stack(prev_act, player.car_id)
                p.extend(list(self.action_stacks[player.car_id]))

        return p

    @staticmethod
    @njit(cache=True)
    def create_car_packet_njit(player_car_position: np.ndarray,
                               player_car_linear_velocity: np.ndarray,
                               car_position: np.ndarray,
                               car_linear_velocity: np.ndarray,
                               car_angular_velocity: np.ndarray,
                               ball_position: np.ndarray,
                               ball_linear_velocity: np.ndarray,
                               fwd: np.ndarray,
                               up: np.ndarray,
                               teammate: bool,
                               boost: float,
                               on_ground: bool,
                               has_jump: bool,
                               has_flip: bool,
                               is_demoed: bool,
                               demo_timer: float,
                               pos_std: int,
                               vel_std: int,
                               ang_std: float,
                               ):
        diff = car_position - player_car_position
        magnitude = np.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)
        car_play_vel = car_linear_velocity - player_car_linear_velocity
        pos_diff = ball_position - car_position
        ball_car_vel = ball_linear_velocity - car_linear_velocity
        return [
            car_position[0] / pos_std, car_position[1] /
            pos_std, car_position[2] / pos_std,
            car_linear_velocity[0] / vel_std, car_linear_velocity[1] / vel_std,
            car_linear_velocity[2] / vel_std,
            car_angular_velocity[0] /
            ang_std, car_angular_velocity[1] / ang_std,
            car_angular_velocity[2] / ang_std,
            diff[0] / pos_std, diff[1] / pos_std, diff[2] / pos_std,
            car_play_vel[0] / vel_std, car_play_vel[1] /
            vel_std, car_play_vel[2] / vel_std,
            pos_diff[0] / pos_std, pos_diff[1] /
            pos_std, pos_diff[2] / pos_std,
            ball_car_vel[0] / vel_std, ball_car_vel[1] /
            vel_std, ball_car_vel[2] / vel_std,
            fwd[0], fwd[1], fwd[2],
            up[0], up[1], up[2],
            boost,
            int(on_ground),
            int(has_flip),
            int(is_demoed),
            int(has_jump),
            magnitude / pos_std,
            int(teammate),
            demo_timer,
        ]

    def create_car_packet(self, player_car: PhysicsObject, car: PhysicsObject,
                          _car: PlayerData, ball: PhysicsObject, teammate: bool):
        diff = car.position - player_car.position
        magnitude = math.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)
        car_play_vel = car.linear_velocity - player_car.linear_velocity
        pos_diff = ball.position - car.position
        ball_car_vel = ball.linear_velocity - car.linear_velocity
        fwd = car.forward()
        up = car.up()
        p = [
            car.position[0] / self.POS_STD, car.position[1] /
            self.POS_STD, car.position[2] / self.POS_STD,
            car.linear_velocity[0] /
            self.VEL_STD, car.linear_velocity[1] / self.VEL_STD,
            car.linear_velocity[2] / self.VEL_STD,
            car.angular_velocity[0] /
            self.ANG_STD, car.angular_velocity[1] / self.ANG_STD,
            car.angular_velocity[2] / self.ANG_STD,
            diff[0] / self.POS_STD, diff[1] /
            self.POS_STD, diff[2] / self.POS_STD,
            car_play_vel[0] / self.VEL_STD, car_play_vel[1] /
            self.VEL_STD, car_play_vel[2] / self.VEL_STD,
            pos_diff[0] / self.POS_STD, pos_diff[1] /
            self.POS_STD, pos_diff[2] / self.POS_STD,
            ball_car_vel[0] / self.VEL_STD, ball_car_vel[1] /
            self.VEL_STD, ball_car_vel[2] / self.VEL_STD,
            fwd[0], fwd[1], fwd[2],
            up[0], up[1], up[2],
            _car.boost_amount,
            int(_car.on_ground),
            int(_car.has_flip),
            int(_car.is_demoed),
            int(_car.has_jump),
            magnitude / self.POS_STD,
            int(teammate),
            self.demo_timers[_car.car_id] / self.DEMO_TIMER_STD,
        ]
        return p

    def create_boost_packet(self, player_car: PhysicsObject, boost_index: int, inverted: bool):
        return NotImplementedError
        # for each boost give the direction, distance, and availability of boost
        boost_avail_list = self.inverted_boosts_availability if inverted else self.boosts_availability
        location = self.inverted_boost_locations[boost_index] if inverted else self.boost_locations[boost_index]
        dist = location - player_car.position
        mag = math.sqrt(dist[0] ** 2 + dist[1] ** 2 +
                        dist[2] ** 2) / self.POS_STD
        val = 0 if not bool(boost_avail_list[boost_index]) else (
            1.0 if location[2] == 73.0 else 0.12)
        p = [
            dist[0] / self.POS_STD, dist[1] /
            self.POS_STD, dist[2] / self.POS_STD,
            val,
            mag
        ]
        return p

    @staticmethod
    @njit(cache=True)
    def add_boosts_to_obs_njit(player_car_position: np.ndarray,
                               boost_avail_list: np.ndarray,
                               location: np.ndarray,
                               boost_values: np.ndarray,
                               pos_std: int):

        dist = location - player_car_position
        dist_std = dist / pos_std

        mag = np.empty(dist.shape[0])
        for i in range(dist.shape[0]):
            mag[i] = np.sqrt(dist[i, 0] * dist[i, 0] + dist[i, 1]
                             * dist[i, 1] + dist[i, 2] * dist[i, 2]) / pos_std
        val = boost_avail_list * boost_values
        return np.column_stack((dist_std, val, mag)).flatten()

    def add_boosts_to_obs(self, obs, player_car: PhysicsObject, inverted: bool):
        boost_avail_list = self.inverted_boosts_availability if inverted else self.boosts_availability
        location = self.inverted_boost_locations if inverted else self.boost_locations
        dist = location - player_car.position
        dist_std = dist / self.POS_STD
        mag = np.linalg.norm(dist, axis=1) / self.POS_STD
        val = boost_avail_list * self.boost_values
        obs.extend(np.column_stack((dist_std, val, mag)).flatten())

        # for i in range(self.boost_locations.shape[0]):
        #     obs.extend(self.create_boost_packet(player_car, i, inverted))

    def add_players_to_obs(self, obs: List, state: GameState, player: PlayerData, ball: PhysicsObject,
                           prev_act: np.ndarray, inverted: bool, previous_model_action, zero_other_players: bool,
                           zero_boost: bool):

        # player_data = self.create_player_packet(player, player.inverted_car_data
        #             if inverted else player.car_data, ball, prev_act, previous_model_action)
        demo_timer = self.demo_timers[player.car_id] / self.DEMO_TIMER_STD
        cid = player.car_id
        if self.any_timers:
            player_data = self.create_player_packet_timers_njit(
                player.inverted_car_data.position if inverted else player.car_data.position,
                player.inverted_car_data.linear_velocity if inverted else player.car_data.linear_velocity,
                player.inverted_car_data.angular_velocity if inverted else player.car_data.angular_velocity,
                player.inverted_car_data.forward() if inverted else player.car_data.forward(),
                player.inverted_car_data.up() if inverted else player.inverted_car_data.up(),
                0 if zero_boost else player.boost_amount, player.on_ground, player.has_jump, player.has_flip,
                player.is_demoed, demo_timer, self.POS_STD, self.VEL_STD, self.ANG_STD,
                ball.position, ball.linear_velocity, prev_act, self.boosttimes[cid],
                self.jumptimes[cid], self.airtimes[cid], self.fliptimes[cid], self.handbrakes[cid],
                self.flipdirs[cid][0], self.flipdirs[cid][1]
            )
            if not self.flip_dir:
                player_data = player_data[:-2]
        else:
            player_data = self.create_player_packet_njit(
                player.inverted_car_data.position if inverted else player.car_data.position,
                player.inverted_car_data.linear_velocity if inverted else player.car_data.linear_velocity,
                player.inverted_car_data.angular_velocity if inverted else player.car_data.angular_velocity,
                player.inverted_car_data.forward() if inverted else player.car_data.forward(),
                player.inverted_car_data.up() if inverted else player.inverted_car_data.up(),
                0 if zero_boost else player.boost_amount, player.on_ground, player.has_jump, player.has_flip,
                player.is_demoed, demo_timer, self.POS_STD, self.VEL_STD, self.ANG_STD,
                ball.position, ball.linear_velocity, prev_act
            )

        if self.stack_size != 0:
            if self.selector:
                self.model_add_action_to_stack(
                    previous_model_action, player.car_id)
                player_data.extend(
                    list(self.model_action_stacks[player.car_id]))

            else:
                self.add_action_to_stack(prev_act, player.car_id)
                player_data.extend(list(self.action_stacks[player.car_id]))

        a_max = 2
        o_max = 3
        a_count = 0
        o_count = 0
        allies = []
        opponents = []
        closest = 0
        if self.only_closest_opp:
            tmp_oppo = [
                p for p in state.players if p.team_num != player.team_num]
            tmp_oppo.sort(key=lambda p: np.linalg.norm(p.inverted_car_data.position if inverted else p.car_data.position
                                                                                                     - player.inverted_car_data.position if inverted else
            player.car_data.position))
            closest = tmp_oppo[0].car_id

        if self.override_cars:
            tmp_oppo = [
                p for p in state.players if p.team_num != player.team_num]
            tmp_oppo.sort(key=lambda p: np.linalg.norm(
                p.car_data.position - player.car_data.position))
            vec = player.car_data.position - ball.position
            vec = vec / np.linalg.norm(vec)
            # put car 400 behind player on vector from ball to player
            new_points = (400 * vec) + player.car_data.position
            new_points[0] = np.clip(new_points[0], -4096, 4096)
            new_points[1] = np.clip(new_points[1], -5120, 5120)
            new_points[2] = np.clip(new_points[2], 0, 2000)
            p = copy.deepcopy(tmp_oppo[0])  # testing imbalance issue
            p.car_data.position = new_points
            # opponents.append(self.create_car_packet(player.inverted_car_data if inverted else player.car_data,
            #                                         p.inverted_car_data if inverted else p.car_data, p, ball,
            #                                         p.team_num == player.team_num))
            demo_timer = self.demo_timers[p.car_id] / self.DEMO_TIMER_STD
            opponents.append(
                self.create_car_packet_njit(player.inverted_car_data.position if inverted else player.car_data.position,
                                            player.inverted_car_data.linear_velocity if inverted else player.car_data.linear_velocity,
                                            p.inverted_car_data.position if inverted else p.car_data.position,
                                            p.inverted_car_data.linear_velocity if inverted else p.car_data.linear_velocity,
                                            p.inverted_car_data.angular_velocity if inverted else p.car_data.angular_velocity,
                                            ball.position, ball.linear_velocity,
                                            p.inverted_car_data.forward() if inverted else p.car_data.forward(),
                                            p.inverted_car_data.up() if inverted else p.car_data.up(),
                                            p.team_num == player.team_num,
                                            p.boost_amount, p.on_ground, p.has_jump, p.has_flip, p.is_demoed,
                                            demo_timer, self.POS_STD, self.VEL_STD, self.ANG_STD
                                            ))
            o_count += 1

        for p in state.players:
            if p.car_id == player.car_id or zero_other_players or self.override_cars:
                continue

            if p.team_num == player.team_num and a_count < a_max:
                a_count += 1
            elif o_count < o_max:
                o_count += 1
            else:
                continue

            if p.team_num == player.team_num and not self.only_closest_opp:
                demo_timer = self.demo_timers[p.car_id] / self.DEMO_TIMER_STD
                allies.append(self.create_car_packet_njit(
                    player.inverted_car_data.position if inverted else player.car_data.position,
                    player.inverted_car_data.linear_velocity if inverted else player.car_data.linear_velocity,
                    p.inverted_car_data.position if inverted else p.car_data.position,
                    p.inverted_car_data.linear_velocity if inverted else p.car_data.linear_velocity,
                    p.inverted_car_data.angular_velocity if inverted else p.car_data.angular_velocity,
                    ball.position, ball.linear_velocity,
                    p.inverted_car_data.forward() if inverted else p.car_data.forward(),
                    p.inverted_car_data.up() if inverted else p.car_data.up(),
                    p.team_num == player.team_num,
                    p.boost_amount, p.on_ground, p.has_jump, p.has_flip, p.is_demoed,
                    demo_timer, self.POS_STD, self.VEL_STD, self.ANG_STD
                ))
            elif not self.only_closest_opp or closest == p.car_id:
                demo_timer = self.demo_timers[p.car_id] / self.DEMO_TIMER_STD
                opponents.append(self.create_car_packet_njit(
                    player.inverted_car_data.position if inverted else player.car_data.position,
                    player.inverted_car_data.linear_velocity if inverted else player.car_data.linear_velocity,
                    p.inverted_car_data.position if inverted else p.car_data.position,
                    p.inverted_car_data.linear_velocity if inverted else p.car_data.linear_velocity,
                    p.inverted_car_data.angular_velocity if inverted else p.car_data.angular_velocity,
                    ball.position, ball.linear_velocity,
                    p.inverted_car_data.forward() if inverted else p.car_data.forward(),
                    p.inverted_car_data.up() if inverted else p.car_data.up(),
                    p.team_num == player.team_num,
                    p.boost_amount, p.on_ground, p.has_jump, p.has_flip, p.is_demoed,
                    demo_timer, self.POS_STD, self.VEL_STD, self.ANG_STD
                ))
            else:
                continue

        if self.only_closest_opp:
            a_count = 0
            o_count = 1

        for _ in range(a_max - a_count):
            allies.append(self.dummy_tm8)

        for _ in range(o_max - o_count):
            opponents.append(self.dummy_player)

        if not self.embed_players:
            random.shuffle(allies)
            random.shuffle(opponents)

        obs.extend(allies)
        obs.extend(opponents)

        return player_data

    def add_action_to_stack(self, new_action: np.ndarray, car_id: int):
        stack = self.action_stacks[car_id]
        stack[self.action_size:] = stack[:-self.action_size]
        stack[:self.action_size] = new_action

    def model_add_action_to_stack(self, new_action: np.ndarray, car_id: int):
        stack = self.model_action_stacks[car_id]
        stack.pop(-1)
        stack.insert(0, new_action[0] / self.model_action_size)

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray,
                  previous_model_action: np.ndarray = None, obs_info=None, zero_boost: bool = False,) -> Any:

        if self.any_timers and obs_info is None:  # if it's not None, this is handled in the obs_info step in the worker
            self._update_addl_timers(player, state, previous_action)
            self.prev_prev_actions[player.car_id] = previous_action  # noqa

        if obs_info is not None:
            # unpack, I'm sure there's a cooler way to do this
            self.boost_timers = obs_info.boost_timers
            self.inverted_boost_timers = obs_info.inverted_boost_timers
            self.boosts_availability = obs_info.boosts_availability
            self.inverted_boosts_availability = obs_info.inverted_boosts_availability
            self.blue_obs = obs_info.blue_obs
            self.orange_obs = obs_info.orange_obs
            self.demo_timers = obs_info.demo_timers
            self.boosttimes = obs_info.boosttimes
            self.jumptimes = obs_info.jumptimes
            self.fliptimes = obs_info.fliptimes
            self.has_flippeds = obs_info.has_flippeds
            self.has_doublejumpeds = obs_info.has_doublejumpeds
            self.flipdirs = obs_info.flipdirs
            self.airtimes = obs_info.airtimes
            self.on_grounds = obs_info.on_grounds
            self.is_jumpings = obs_info.is_jumpings
            self.has_jumpeds = obs_info.has_jumpeds
            self.handbrakes = obs_info.handbrakes

        if player.team_num == 1:
            inverted = True
            ball = state.inverted_ball
        else:
            inverted = False
            ball = state.ball

        if self.end_object is not None and \
                not (self.end_object.position[0] == self.end_object.position[1] == self.end_object.position[2] == -1):
            ball.position[0] = self.end_object.position[0]
            ball.position[1] = self.end_object.position[1]
            ball.position[2] = self.end_object.position[2]
            ball.linear_velocity = np.asarray([0, 0, 0])
            ball.angular_velocity = np.asarray([0, 0, 0])

        obs = []
        players_data = []
        player_dat = self.add_players_to_obs(players_data, state, player, ball, previous_action, inverted,
                                             previous_model_action, self.zero_other_cars, zero_boost)
        obs.extend(player_dat)
        obs.extend(self.create_ball_packet(ball))
        if not self.embed_players and not self.remove_other_cars:
            for p in players_data:
                obs.extend(p)
        # this adds boost timers and direction/distance to all boosts
        # unnecessary if only doing aerial stuff
        if self.extra_boost_info:
            if inverted:
                obs.extend(self.orange_obs)
            else:
                obs.extend(self.blue_obs)
            # self.add_boosts_to_obs(obs, player.inverted_car_data if inverted else player.car_data, inverted)
            obs.extend(
                self.add_boosts_to_obs_njit(player.inverted_car_data.position if inverted else player.car_data.position,
                                            self.inverted_boosts_availability if inverted else self.boosts_availability,
                                            self.inverted_boost_locations if inverted else self.boost_locations,
                                            self.boost_values, self.POS_STD))
        if self.expanding and not self.embed_players:
            return np.expand_dims(np.fromiter(obs, dtype=np.float32, count=len(obs)), 0)
            # return torch.FloatTensor([obs])
            # return np.expand_dims(obs, 0)
        elif self.expanding and self.embed_players:
            return np.expand_dims(np.fromiter(obs, dtype=np.float32, count=len(obs)), 0), \
                   np.asarray([players_data])
            # return torch.FloatTensor([obs]), torch.FloatTensor([players_data])
            # return np.expand_dims(obs, 0), np.expand_dims(players_data, 0)
        elif not self.expanding and not self.embed_players:
            return obs
        else:
            return obs, players_data

    def get_obs_space(self) -> Space:
        players = self.num_players - 1 or 5
        car_size = len(self.dummy_player)
        player_size = 251 + (self.stack_size * self.action_size)
        return Tuple((
            Box(-np.inf, np.inf, (1, player_size)),
            Box(-np.inf, np.inf, (1, players, car_size)),
        ))


class CoyoteStackedObsBuilder(CoyoteObsBuilder):
    def __init__(self, stack_size: int = 5, action_parser=None
                 ):
        return NotImplemented
        super().__init__()
        self.action_parser = action_parser
        self.stack_size = stack_size

    def reset(self, initial_state: GameState):
        super().reset(initial_state)
        self.action_stacks = {}
        for p in initial_state.players:
            self.action_stacks[p.car_id] = np.concatenate(
                [self.default_action] * self.stack_size)
        self.action_parser.reset(initial_state)


class CoyoteObsBuilder_Legacy(ObsBuilder):
    def __init__(self, tick_skip=8, team_size=3, expanding: bool = True, extra_boost_info: bool = True,
                 embed_players=False, stack_size=0, action_parser=None, env: Gym = None, infinite_boost_odds=0,
                 only_closest_opp=False,
                 selector=False,
                 end_object_choice=None,
                 remove_other_cars=False,
                 zero_other_cars=False,
                 obs_info=None,
                 ):
        """use this when it was trained before the NJIT changes around 1/12/2023"""
        super().__init__()
        self.obs_info = obs_info
        self.zero_other_cars = zero_other_cars
        self.remove_other_cars = remove_other_cars
        self.expanding = expanding
        self.only_closest_opp = only_closest_opp
        self.extra_boost_info = extra_boost_info
        self.POS_STD = 2300
        self.VEL_STD = 2300
        self.ANG_STD = 5.5
        self.BOOST_TIMER_STD = 10
        self.DEMO_TIMER_STD = 3
        self.dummy_player = [0] * 35
        self.boost_locations = np.array(BOOST_LOCATIONS)
        self.inverted_boost_locations = self.boost_locations[::-1]
        self.boost_timers = np.zeros(self.boost_locations.shape[0])
        self.inverted_boost_timers = np.zeros(self.boost_locations.shape[0])
        self.boosts_availability = np.zeros(self.boost_locations.shape[0])
        self.inverted_boosts_availability = np.zeros(self.boost_locations.shape[0])
        self.boost_objs = []
        self.inverted_boost_objs = []
        self.state = None
        self.time_interval = tick_skip / 120
        self.demo_timers = None
        self.num_players = team_size * 2
        self.generic_obs = None
        self.blue_obs = None
        self.orange_obs = None
        self.embed_players = embed_players
        self.selector = selector
        self.default_action = np.zeros(8)
        self.stack_size = stack_size
        self.action_stacks = {}
        self.model_action_stacks = {}
        self.action_size = self.default_action.shape[0]
        self.action_parser = action_parser
        if self.action_parser is not None:
            self.model_action_size = action_parser.get_model_action_size()
        self.env = env
        self.infinite_boost_odds = infinite_boost_odds
        self.infinite_boost_episode = False
        self.end_object_choice = end_object_choice
        self.end_object_tracker = 0
        if end_object_choice is not None and end_object_choice == "random":
            self.end_object_tracker = 0
        elif end_object_choice is not None:
            self.end_object_tracker = int(self.end_object_choice)
        self.big_boosts = [BOOST_LOCATIONS[i] for i in [3, 4, 15, 18, 29, 30]]
        self.big_boosts = np.asarray(self.big_boosts)
        self.big_boosts[:, -1] = 18

    def reset(self, initial_state: GameState):
        self.state = None
        if self.obs_info is None:
            self.boost_timers = np.zeros(self.boost_locations.shape[0])
            self.inverted_boost_timers = np.zeros(
                self.boost_locations.shape[0])
            self.demo_timers = np.zeros(
                max(p.car_id for p in initial_state.players) + 1)
            self.blue_obs = []
            self.orange_obs = []

        self.action_stacks = {}
        if self.stack_size != 0 and not self.selector:
            for p in initial_state.players:
                self.action_stacks[p.car_id] = np.concatenate(
                    [self.default_action] * self.stack_size)

        self.model_action_stacks = {}
        if self.stack_size != 0 and self.selector:
            for p in initial_state.players:
                self.model_action_stacks[p.car_id] = [0.] * self.stack_size

        if self.action_parser is not None:
            self.action_parser.reset(initial_state)

        if self.env is not None:
            if random.random() <= self.infinite_boost_odds:
                self.env.update_settings(boost_consumption=0)
                self.infinite_boost_episode = True
            else:
                self.env.update_settings(boost_consumption=1)
                self.infinite_boost_episode = False

    def pre_step(self, state: GameState):
        self.state = state
        # create player/team agnostic items (do these even exist?)
        self._update_timers(state)
        # create team specific things
        self.blue_obs = self.boost_timers / self.BOOST_TIMER_STD
        self.orange_obs = self.inverted_boost_timers / self.BOOST_TIMER_STD

        if self.env is not None:
            if self.infinite_boost_episode:
                for player in state.players:
                    player.boost_amount = 2
            else:
                for player in state.players:
                    player.boost_amount /= 1

    def _update_timers(self, state: GameState):
        current_boosts = state.boost_pads
        boost_locs = self.boost_locations
        demo_states = [[p.car_id, p.is_demoed] for p in state.players]

        for i in range(len(current_boosts)):
            if current_boosts[i] == self.boosts_availability[i]:
                if self.boosts_availability[i] == 0:
                    self.boost_timers[i] = max(
                        0, self.boost_timers[i] - self.time_interval)
            else:
                if self.boosts_availability[i] == 0:
                    self.boosts_availability[i] = 1
                    self.boost_timers[i] = 0
                else:
                    self.boosts_availability[i] = 0
                    if boost_locs[i][2] == 73:
                        self.boost_timers[i] = 10.0
                    else:
                        self.boost_timers[i] = 4.0
        self.boosts_availability = current_boosts
        self.inverted_boost_timers = self.boost_timers[::-1]
        self.inverted_boosts_availability = self.boosts_availability[::-1]

        for cid, dm in demo_states:
            if dm == True:  # Demoed
                prev_timer = self.demo_timers[cid]
                if prev_timer > 0:
                    self.demo_timers[cid] = max(
                        0, prev_timer - self.time_interval)
                else:
                    self.demo_timers[cid] = 3
            else:  # Not demoed
                self.demo_timers[cid] = 0

    def create_ball_packet(self, ball: PhysicsObject):
        p = [
            ball.position[0] / self.POS_STD, ball.position[1] / self.POS_STD, ball.position[2] / self.POS_STD,
            ball.linear_velocity[0] / self.VEL_STD, ball.linear_velocity[1] / self.VEL_STD, ball.linear_velocity[2] / self.VEL_STD,
            ball.angular_velocity[0] / self.ANG_STD, ball.angular_velocity[1] / self.ANG_STD, ball.angular_velocity[2] / self.ANG_STD,
            math.sqrt(ball.linear_velocity[0] ** 2 + ball.linear_velocity[1] ** 2 + ball.linear_velocity[2] ** 2)/2300,
            int(ball.position[2] <= 100),
        ]
        return p

    def create_player_packet(self, player: PlayerData, car: PhysicsObject, ball: PhysicsObject, prev_act: np.ndarray,
                             prev_model_act: np.ndarray, zero_boost: bool):
        pos_diff = ball.position - car.position
        vel_diff = ball.linear_velocity - car.linear_velocity
        fwd = car.forward()
        up = car.up()
        p = [
            car.position[0] / self.POS_STD, car.position[1] / self.POS_STD, car.position[2] / self.POS_STD,
            car.linear_velocity[0] / self.VEL_STD, car.linear_velocity[1] / self.VEL_STD, car.linear_velocity[2] / self.VEL_STD,
            car.angular_velocity[0] / self.ANG_STD, car.angular_velocity[1] / self.ANG_STD, car.angular_velocity[2] / self.ANG_STD,
            pos_diff[0] / self.POS_STD, pos_diff[1] / self.POS_STD, pos_diff[2] / self.POS_STD,
            vel_diff[0] / self.VEL_STD, vel_diff[1] / self.VEL_STD, vel_diff[2] / self.VEL_STD,
            fwd[0], fwd[1], fwd[2],
            up[0], up[1], up[2],
            math.sqrt(car.linear_velocity[0] ** 2 + car.linear_velocity[1] ** 2 + car.linear_velocity[2] ** 2)/2300,
            0 if zero_boost else player.boost_amount,
            int(player.on_ground),
            int(player.has_flip),
            int(player.is_demoed),
            int(player.has_jump),
            self.demo_timers[player.car_id] / self.DEMO_TIMER_STD,
            prev_act[0], prev_act[1], prev_act[2], prev_act[3], prev_act[4], prev_act[5], prev_act[6], prev_act[7],
        ]
        if self.stack_size != 0:
            if self.selector:
                self.model_add_action_to_stack(prev_model_act, player.car_id)
                p.extend(list(self.model_action_stacks[player.car_id]))

            else:
                self.add_action_to_stack(prev_act, player.car_id)
                p.extend(list(self.action_stacks[player.car_id]))

        return p

    def create_car_packet(self, player_car: PhysicsObject, car: PhysicsObject,
                          _car: PlayerData, ball: PhysicsObject, teammate: bool):
        diff = car.position - player_car.position
        magnitude = math.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)
        car_play_vel = car.linear_velocity - player_car.linear_velocity
        pos_diff = ball.position - car.position
        ball_car_vel = ball.linear_velocity - car.linear_velocity
        fwd = car.forward()
        up = car.up()
        p = [
                car.position[0] / self.POS_STD, car.position[1] / self.POS_STD, car.position[2] / self.POS_STD,
                car.linear_velocity[0] / self.VEL_STD, car.linear_velocity[1] / self.VEL_STD, car.linear_velocity[2] / self.VEL_STD,
                car.angular_velocity[0] / self.ANG_STD, car.angular_velocity[1] / self.ANG_STD, car.angular_velocity[2] / self.ANG_STD,
                diff[0] / self.POS_STD, diff[1] / self.POS_STD, diff[2] / self.POS_STD,
                car_play_vel[0] / self.VEL_STD, car_play_vel[1] / self.VEL_STD, car_play_vel[2] / self.VEL_STD,
                pos_diff[0] / self.POS_STD, pos_diff[1] / self.POS_STD, pos_diff[2] / self.POS_STD,
                ball_car_vel[0] / self.VEL_STD, ball_car_vel[1] / self.VEL_STD, ball_car_vel[2] / self.VEL_STD,
                fwd[0], fwd[1], fwd[2],
                up[0], up[1], up[2],
                _car.boost_amount,
                int(_car.on_ground),
                int(_car.has_flip),
                int(_car.is_demoed),
                int(_car.has_jump),
                magnitude/self.POS_STD,
                teammate,
                self.demo_timers[_car.car_id] / self.DEMO_TIMER_STD,
            ]
        return p

    def create_boost_packet(self, player_car: PhysicsObject, boost_index: int, inverted: bool):
        # for each boost give the direction, distance, and availability of boost
        boost_avail_list = self.inverted_boosts_availability if inverted else self.boosts_availability
        location = self.inverted_boost_locations[boost_index] if inverted else self.boost_locations[boost_index]
        dist = location - player_car.position
        mag = math.sqrt(dist[0] ** 2 + dist[1] ** 2 + dist[2] ** 2) / self.POS_STD
        val = 0 if not bool(boost_avail_list[boost_index]) else (1.0 if location[2] == 73.0 else 0.12)
        p = [
            dist[0] / self.POS_STD, dist[1] / self.POS_STD, dist[2] / self.POS_STD,
            val,
            mag
        ]
        return p

    def add_boosts_to_obs(self, obs, player_car: PhysicsObject, inverted: bool):
        for i in range(self.boost_locations.shape[0]):
            obs.extend(self.create_boost_packet(player_car, i, inverted))

    def add_players_to_obs(self, obs: List, state: GameState, player: PlayerData, ball: PhysicsObject,
                           prev_act: np.ndarray, inverted: bool, previous_model_action, zero_other_players: bool,
                           zero_boost: bool):

        player_data = self.create_player_packet(player, player.inverted_car_data
                                                if inverted else player.car_data, ball, prev_act, previous_model_action,
                                                zero_boost)
        a_max = 2
        o_max = 3
        a_count = 0
        o_count = 0
        allies = []
        opponents = []
        closest = 0
        if self.only_closest_opp:
            tmp_oppo = [
                p for p in state.players if p.team_num != player.team_num]
            tmp_oppo.sort(key=lambda p: np.linalg.norm(p.inverted_car_data.position if inverted else p.car_data.position
                                                                                                     - player.inverted_car_data.position if inverted else
            player.car_data.position))
            closest = tmp_oppo[0].car_id

        for p in state.players:
            if p.car_id == player.car_id or zero_other_players:
                continue

            if p.team_num == player.team_num and a_count < a_max:
                a_count += 1
            elif o_count < o_max:
                o_count += 1
            else:
                continue

            if p.team_num == player.team_num and not self.only_closest_opp:
                allies.append(self.create_car_packet(player.inverted_car_data if inverted else player.car_data,
                              p.inverted_car_data if inverted else p.car_data, p, ball, p.team_num == player.team_num))
            elif not self.only_closest_opp or closest == p.car_id:
                opponents.append(self.create_car_packet(player.inverted_car_data if inverted else player.car_data,
                                                        p.inverted_car_data if inverted else p.car_data, p, ball,
                                                        p.team_num == player.team_num))
            else:
                continue

        if self.only_closest_opp:
            a_count = 0
            o_count = 1

        for _ in range(a_max - a_count):
            allies.append(self.dummy_player)

        for _ in range(o_max - o_count):
            opponents.append(self.dummy_player)

        if not self.embed_players:
            random.shuffle(allies)
            random.shuffle(opponents)

        obs.extend(allies)
        obs.extend(opponents)

        return player_data

    def add_action_to_stack(self, new_action: np.ndarray, car_id: int):
        stack = self.action_stacks[car_id]
        stack[self.action_size:] = stack[:-self.action_size]
        stack[:self.action_size] = new_action

    def model_add_action_to_stack(self, new_action: np.ndarray, car_id: int):
        stack = self.model_action_stacks[car_id]
        stack.pop(-1)
        stack.insert(0, new_action[0] / self.model_action_size)

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray,
                  previous_model_action: np.ndarray = None, obs_info=None, zero_boost: bool = False) -> Any:

        if obs_info is not None:
            # unpack, I'm sure there's a cooler way to do this
            self.boost_timers = obs_info.boost_timers
            self.inverted_boost_timers = obs_info.inverted_boost_timers
            self.boosts_availability = obs_info.boosts_availability
            self.inverted_boosts_availability = obs_info.inverted_boosts_availability
            self.blue_obs = obs_info.blue_obs
            self.orange_obs = obs_info.orange_obs
            self.demo_timers = obs_info.demo_timers

        if player.team_num == 1:
            inverted = True
            ball = state.inverted_ball
        else:
            inverted = False
            ball = state.ball

        obs = []
        players_data = []
        player_dat = self.add_players_to_obs(players_data, state, player, ball, previous_action, inverted,
                                             previous_model_action, self.zero_other_cars, zero_boost)
        obs.extend(player_dat)
        obs.extend(self.create_ball_packet(ball))
        if not self.embed_players and not self.remove_other_cars:
            for p in players_data:
                obs.extend(p)
        # this adds boost timers and direction/distance to all boosts
        # unnecessary if only doing aerial stuff
        if self.extra_boost_info:
            if inverted:
                obs.extend(self.orange_obs)
            else:
                obs.extend(self.blue_obs)
            self.add_boosts_to_obs(obs, player.inverted_car_data if inverted else player.car_data, inverted)
        if self.expanding and not self.embed_players:
            return np.expand_dims(np.fromiter(obs, dtype=np.float32, count=len(obs)), 0)
            # return torch.FloatTensor([obs])
            # return np.expand_dims(obs, 0)
        elif self.expanding and self.embed_players:
            return np.expand_dims(np.fromiter(obs, dtype=np.float32, count=len(obs)), 0), \
                   np.asarray([players_data])
            # return torch.FloatTensor([obs]), torch.FloatTensor([players_data])
            # return np.expand_dims(obs, 0), np.expand_dims(players_data, 0)
        elif not self.expanding and not self.embed_players:
            return obs
        else:
            return obs, players_data


if __name__ == "__main__":
    print("nope")
