from typing import Any

import gym.spaces
import numpy as np
from gym.spaces import Discrete
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.gamestates import GameState

# from rlgym_tools.extra_action_parsers.kbm_act import KBMAction

# NectoActionOLD = KBMAction


class CoyoteAction(ActionParser):
    def __init__(self, version=None):
        super().__init__()
        self._lookup_table = self.make_lookup_table(version)

    @staticmethod
    def make_lookup_table(version):
        actions = []
        if version is None or version == "Normal":
            # Ground
            for throttle in (-1, 0, 0.5, 1):
                for steer in (-1, -0.5, 0, 0.5, 1):
                    for boost in (0, 1):
                        for handbrake in (0, 1):
                            if boost == 1 and throttle != 1:
                                continue
                            actions.append(
                                [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
            # Aerial
            for pitch in (-1, -0.75, -0.5, 0, 0.5, 0.75, 1):
                for yaw in (-1, -0.75, -0.5, 0, 0.5, 0.75, 1):
                    for roll in (-1, 0, 1):
                        for jump in (0, 1):
                            for boost in (0, 1):
                                if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                    continue
                                if pitch == roll == jump == 0:  # Duplicate with ground
                                    continue
                                # Enable handbrake for potential wavedashes
                                handbrake = jump == 1 and (
                                        pitch != 0 or yaw != 0 or roll != 0)
                                actions.append(
                                    [boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
            # append stall
            actions.append([0, 1, 0, 0, -1, 1, 0, 0])
            actions = np.array(actions)
        return actions

    def get_action_space(self) -> gym.spaces.Space:
        return Discrete(len(self._lookup_table))

    @staticmethod
    def get_model_action_space() -> int:
        return 1

    def get_model_action_size(self) -> int:
        return len(self._lookup_table)

    def parse_actions(self, actions: Any, state: GameState, zero_boost: bool = False) -> np.ndarray:
        # hacky pass through to allow multiple types of agent actions while still parsing nectos

        # strip out fillers, pass through 8sets, get look up table values, recombine
        parsed_actions = []
        for action in actions:
            # test
            # parsed_actions.append([0, 0, 0, 0, 0, 0, 0, 0])
            # continue
            # support reconstruction
            if action.size != 8:
                if action.shape == 0:
                    action = np.expand_dims(action, axis=0)
                # to allow different action spaces, pad out short ones (assume later unpadding in parser)
                action = np.pad(action.astype(
                    'float64'), (0, 8 - action.size), 'constant', constant_values=np.NAN)

            if np.isnan(action).any():  # it's been padded, delete to go back to original
                stripped_action = (
                    action[~np.isnan(action)]).squeeze().astype('int')
                parsed_actions.append(self._lookup_table[stripped_action])
            else:
                parsed_actions.append(action)
        if zero_boost:
            parsed_actions[6] = 0
        return np.asarray(parsed_actions)
