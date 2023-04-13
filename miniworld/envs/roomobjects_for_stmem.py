import math

from gymnasium import utils

from miniworld.entity import COLOR_NAMES, Ball, Box, Key
from miniworld.miniworld import MiniWorldEnv

import numpy as np


class RoomObjectsSTMEM(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Single room with multiple objects. Inspired by the single room environment
    of the Generative Query Networks paper:
    https://deepmind.com/blog/neural-scene-representation-and-rendering/

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |
    | 3   | move back                   |
    | 4   | pick up                     |
    | 5   | drop                        |
    | 6   | toggle / activate an object |
    | 7   | complete task               |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    None

    ## Arguments

    ```python
    RoomObjects(size=16)
    ```

    `size`: size of world

    """

    def __init__(self, size=10, **kwargs):
        assert size >= 2
        self.size = size
        
        self._action_set_of_left_right = {
            "circle_cw": [
                [-1, 0, 0], # move left
                [0, 1, 0], # move up
                [1, 0, 0], # move right
                [0, -1, 0], # move down
            ],
            "circle_ccw": [
                [1, 0, 0], # move right
                [0, 1, 0], # move up
                [-1, 0, 0], # move left
                [0, -1, 0], # move down
            ],
            "up_and_down": [
                [0, 1, 0], # move up
                [0, -1, 0], # move down
            ],
            "left_and_right": [
                [-1, 0, 0], # move left
                [1, 0, 0], # move right
                [1, 0, 0], # move right
                [-1, 0, 0], # move left
            ],
        }
        
        self._action_set_of_front_back = {
            "circle_cw": [
                [0, 0, -1], # move left
                [0, 1, 0], # move up
                [0, 0, 1], # move right
                [0, -1, 0], # move down
            ],
            "circle_ccw": [
                [0, 0, 1], # move right
                [0, 1, 0], # move up
                [0, 0, -1], # move left
                [0, -1, 0], # move down
            ],
            "up_and_down": [
                [0, 1, 0], # move up
                [0, -1, 0], # move down
            ],
            "left_and_right": [
                [0, 0, -1], # move left
                [0, 0, 1], # move right
                [0, 0, 1], # move right
                [0, 0, -1], # move left
            ],
        }


        MiniWorldEnv.__init__(self, max_episode_steps=math.inf, **kwargs)
        utils.EzPickle.__init__(self, size, **kwargs)
        
    def _gen_world(self):
        self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )

        # Reduce chances that objects are too close to see
        self.agent.radius = 5.0
        colorlist = list(COLOR_NAMES)
       
        self._actions = [] 
        # object on the front
        self.place_entity(
            Ball(color=colorlist[self.np_random.choice(len(colorlist))], size=0.9),
            pos=[self.size/2+4.5, 0, self.size/2]
        )
        self._actions.append(self._action_set_of_front_back[self.np_random.choice(list(self._action_set_of_front_back.keys()))])
        # object on the left
        self.place_entity(
            Ball(color=colorlist[self.np_random.choice(len(colorlist))], size=0.9),
            pos=[self.size/2, 0, self.size/2-4.5]
        )
        self._actions.append(self._action_set_of_left_right[self.np_random.choice(list(self._action_set_of_left_right.keys()))])
        # object on the right
        self.place_entity(
            Ball(color=colorlist[self.np_random.choice(len(colorlist))], size=0.9),
            pos=[self.size/2, 0, self.size/2+4.5]
        )
        self._actions.append(self._action_set_of_left_right[self.np_random.choice(list(self._action_set_of_left_right.keys()))])
         # object on the back
        self.place_entity(
            Ball(color=colorlist[self.np_random.choice(len(colorlist))], size=0.9),
            pos=[self.size/2-4.5, 0, self.size/2]
        )
        self._actions.append(self._action_set_of_front_back[self.np_random.choice(list(self._action_set_of_front_back.keys()))])
     
        self._action_idxs = [0]*len(self._actions)

        self.place_agent(dir=0.785, pos=[self.size/2, 0, self.size/2]) # start from 45 angle

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        for i in range(len(self._actions)):
            self.entities[i].pos = [x + y for x, y in zip(self.entities[i].pos, self._actions[i][self._action_idxs[i]])]
            self._action_idxs[i] = (self._action_idxs[i] + 1) % len(self._actions[i])
        return obs, reward, termination, truncation, info
