import math

from gymnasium import utils

from miniworld.entity import COLOR_NAMES, Ball, Box, Key
from miniworld.miniworld import MiniWorldEnv

import numpy as np
from itertools import permutations

obs_width=84
obs_height=84
window_width=840
window_height=840

COLORS_FOR_NOISY_TV = {
    "Cyan": (0, 255, 255),
    "Magenta": (255, 0, 255),
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
    "Gray": (128, 128, 128),
    "Brown": (165,42,42),
    "Orange": (255, 165, 0),
    "Purple": (128, 0, 128),
    "Pink": (255, 192, 203),
    "Teal": (0, 128, 128),
    "Lavender": (230, 230, 250),
}

OBJ_TYPES = ["box", "ball", "key"]

ACTION_SET = {
    "circle_cw": [
        [0, 0, -0.5], # move left
        [0, 1, 0], # move up
        [0, 0, 0.5], # move right
        [0, 0, 0.5], # move right
        [0, -1, 0], # move down
        [0, 0, -0.5], # move left
    ],
    "circle_ccw": [
        [0, 0, 0.5], # move right
        [0, 1, 0], # move up
        [0, 0, -0.5], # move left
        [0, 0, -0.5], # move left
        [0, -1, 0], # move down
        [0, 0, 0.5], # move right
    ],
    "up_and_down": [
        [0, 1, 0], # move up
        [0, -1, 0], # move down
        [0, 1, 0], # move up
        [0, -1, 0], # move down
        [0, 1, 0], # move up
        [0, -1, 0], # move down
    ],
    "left_and_right_up_and_down": [
        [0, 0, -0.5], # move left
        [0, 0, 0.5], # move right
        [0, 0, 0.5], # move right
        [0, 0,-0.5], # move left
        [0, 1, 0], # move up
        [0, -1, 0], # move down
    ],
}

class RoomNoisyTVSTMEM(MiniWorldEnv, utils.EzPickle):
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
        
        # noisy TV full set
        self._noisy_tv_color_full_set = []
        for colors in permutations(list(COLORS_FOR_NOISY_TV.keys()), 4):
            self._noisy_tv_color_full_set.append(colors)
        
        # number of objects
        self._num_objs = 8
        
        # min_x, max_x, min_z, max_z        
        self._positions = [
            [self.size/2-4.5, self.size/2-3,   self.size/2+2.5, self.size/2+3.5], # front-left
            [self.size/2-4.5, self.size/2-3,   self.size/2-1.5, self.size/2+1.5], # front
            [self.size/2-4.5, self.size/2-3,   self.size/2-3.5, self.size/2-2.5], # front-right
            [self.size/2-1.5, self.size/2+1.5, self.size/2+3,   self.size/2+4.5], # left
            [self.size/2-1.5, self.size/2+1.5, self.size/2-4.5, self.size/2-3], # right
            [self.size/2+3, self.size/2+4.5,   self.size/2+3,   self.size/2+3.5], # back-left
            [self.size/2+3, self.size/2+4.5,   self.size/2-3.5, self.size/2-3], # back-right
        ]       

        MiniWorldEnv.__init__(self,
                              obs_width=obs_width,
                              obs_height=obs_height,
                              window_width=window_width,
                              window_height=window_height,
                              max_episode_steps=math.inf, **kwargs)
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

        colorlist = list(COLOR_NAMES)

        self._actions = []
        for i in range(self._num_objs):
            _type = np.random.choice(OBJ_TYPES)
            #_position = self._positions[np.random.choice(np.arange(len(self._positions)))]
            _position = self._positions[i%len(self._positions)] # to well distribute objects
            if _type == "ball":
                self.place_entity(Ball(color=colorlist[self.np_random.choice(len(colorlist))], size=0.75), min_x=_position[0], max_x=_position[1], min_z=_position[2], max_z=_position[3])
            elif _type == "box":
                self.place_entity(Box(color=colorlist[self.np_random.choice(len(colorlist))], size=0.75), min_x=_position[0], max_x=_position[1], min_z=_position[2], max_z=_position[3])
            elif _type == "key":
                self.place_entity(Key(color=colorlist[self.np_random.choice(len(colorlist))], size=0.4), min_x=_position[0], max_x=_position[1], min_z=_position[2], max_z=_position[3])
            self._actions.append(ACTION_SET[self.np_random.choice(list(ACTION_SET.keys()))])
        self._action_idxs = [0]*len(self._actions)

        # noise TV entities        
        np.random.shuffle(self._noisy_tv_color_full_set)
        self.place_entity(Box(color=colorlist[-1], size=0.6),
                          pos=[self.size/2+5.2, 0.4, self.size/2-0.3], dir=0)
        self.place_entity(Box(color=colorlist[-2], size=0.6),
                          pos=[self.size/2+5.2, 1.0, self.size/2-0.3], dir=0)
        self.place_entity(Box(color=colorlist[-3], size=0.6),
                          pos=[self.size/2+5.2, 0.4, self.size/2+0.3], dir=0)
        self.place_entity(Box(color=colorlist[-4], size=0.6),
                          pos=[self.size/2+5.2, 1.0, self.size/2+0.3], dir=0)
        self._noisy_tv_color_idx = 0

        #self.place_agent(dir=(np.random.rand()*0.2-0.1), pos=[self.size/2, 0, self.size/2])
        self.place_agent(dir=180*np.pi/180, pos=[self.size/2, 0, self.size/2])

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        
        # move objects
        for i in range(len(self._actions)):
            self.entities[i].pos[0] = self.entities[i].pos[0] + self._actions[i][self._action_idxs[i]][0] * np.cos(self.entities[i].dir) + self._actions[i][self._action_idxs[i]][2] * np.sin(self.entities[i].dir) # originally -sin, but used + in this env
            self.entities[i].pos[1] += self._actions[i][self._action_idxs[i]][1]
            self.entities[i].pos[2] = self.entities[i].pos[2] + self._actions[i][self._action_idxs[i]][0] * np.sin(self.entities[i].dir) + self._actions[i][self._action_idxs[i]][2] * np.cos(self.entities[i].dir)
            self._action_idxs[i] = (self._action_idxs[i] + 1) % len(self._actions[i])
            
        # change noisy TV color
        if self.step_count >= 37:
            for i in range(4):
                self.entities[self._num_objs+i].color_vec = np.array(COLORS_FOR_NOISY_TV[self._noisy_tv_color_full_set[self._noisy_tv_color_idx][i]]) / 255.0
            self._noisy_tv_color_idx  = (self._noisy_tv_color_idx + 1) % len(self._noisy_tv_color_full_set)
        return obs, reward, termination, truncation, info
