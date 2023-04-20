import math

from gymnasium import utils

from miniworld.entity import COLOR_NAMES, Ball, Box, Key, Potion
from miniworld.miniworld import MiniWorldEnv

import numpy as np

obs_width=84
obs_height=84
window_width=840
window_height=840

OBJ_TYPES = ["box", "ball", "key"]
COLORS = {
    "Cyan": (0, 255, 255),
    "Magenta": (255, 0, 255),
    "Gray": (128, 128, 128),
    "Brown": (165,42,42),
    "Orange": (255, 165, 0),
    "Purple": (128, 0, 128),
    "Pink": (255, 192, 203),
    "Teal": (0, 128, 128),
    "Lavender": (230, 230, 250),
}

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
        
        self._action_set = {
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

        # Reduce chances that objects are too close to see
        self.agent.radius = 5.0
        colorlist = list(COLOR_NAMES)
        # [x,y,z,dir] 
        positions = {
            "front":       [self.size/2-4.5, 0, self.size/2,     0*np.pi/180], # front
            "front-left":  [self.size/2-3.5, 0, self.size/2+3.5, 45*np.pi/180], # the front-left corner
            "left":        [self.size/2,     0, self.size/2+4.5, 90*np.pi/180], # left
            "back-left":   [self.size/2+3.5, 0, self.size/2+3.5, 125*np.pi/180], # the back left corner
            "back-right":  [self.size/2+3.5, 0, self.size/2-3.5, 225*np.pi/180], # the back right corner
            "right":       [self.size/2,     0, self.size/2-4.5, 270*np.pi/180], # right
            "front-right": [self.size/2-3.5, 0, self.size/2-3.5, 315*np.pi/180], # the front right corner
            "back":        [self.size/2+4.5, 0, self.size/2,     180*np.pi/180], # the back
            "back_hidden": [self.size/2+4.5, -10, self.size/2,   180*np.pi/180], # the back hidden
        }

        self._actions = [] 
        for pos_name, pos in positions.items():
            dir = pos[3]
            _type = np.random.choice(OBJ_TYPES)
            if _type == "ball":
                self.place_entity(Ball(color=colorlist[self.np_random.choice(len(colorlist))], size=0.75), pos=pos[:3], dir=dir)
            elif _type == "box":
                self.place_entity(Box(color=colorlist[self.np_random.choice(len(colorlist))], size=0.75), pos=pos[:3], dir=dir)
            elif _type == "key":
                self.place_entity(Key(color=colorlist[self.np_random.choice(len(colorlist))], size=0.4), pos=pos[:3], dir=dir)
            self._actions.append(self._action_set[self.np_random.choice(list(self._action_set.keys()))])
        self._action_idxs = [0]*len(self._actions)

        self.place_agent(dir=180*np.pi/180, pos=[self.size/2, 0, self.size/2]) # start from 45 angle

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        if self.step_count == 45:
            self.entities[len(self._actions)-2].pos[1] -= 10 # back
            self.entities[len(self._actions)-1].pos[1] += 10 # back hidden
        for i in range(len(self._actions)):
            self.entities[i].pos[0] = self.entities[i].pos[0] + self._actions[i][self._action_idxs[i]][0] * np.cos(self.entities[i].dir) + self._actions[i][self._action_idxs[i]][2] * np.sin(self.entities[i].dir) # originally -sin, but used + in this env
            self.entities[i].pos[1] += self._actions[i][self._action_idxs[i]][1]
            self.entities[i].pos[2] = self.entities[i].pos[2] + self._actions[i][self._action_idxs[i]][0] * np.sin(self.entities[i].dir) + self._actions[i][self._action_idxs[i]][2] * np.cos(self.entities[i].dir)
            self._action_idxs[i] = (self._action_idxs[i] + 1) % len(self._actions[i])
        return obs, reward, termination, truncation, info

    def action_start(self):
        if self._action_idxs[0] == 0:
            return True
        else:
            return False