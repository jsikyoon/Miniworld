import math

from gymnasium import utils

from miniworld.entity import COLOR_NAMES, Ball, Box, Key, Potion
from miniworld.miniworld import MiniWorldEnv

import numpy as np

obs_width=84
obs_height=84
window_width=840
window_height=840

#OBJ_TYPES = ["box", "ball", "key"]
OBJ_TYPES = ["box", "ball"]
#OBJ_TYPES = ["box"]
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
                [0, 0, -1], # move left
                [0, 1, 0], # move up
                [0, 0, 1], # move right
                [0, 0, 1], # move right
                [0, -1, 0], # move down
                [0, 0, -1], # move left
            ],
            "circle_ccw": [
                [0, 0, 1], # move right
                [0, 1, 0], # move up
                [0, 0, -1], # move left
                [0, 0, -1], # move left
                [0, -1, 0], # move down
                [0, 0, 1], # move right
            ],
            "up_and_down": [
                [0, 1, 0], # move up
                [0, -1, 0], # move down
                [0, 1, 0], # move up
                [0, -1, 0], # move down
                [0, 1, 0], # move up
                [0, -1, 0], # move down
            ],
            "left_and_right": [
                [0, 0, -1], # move left
                [0, 0, 1], # move right
                [0, 0, 1], # move right
                [0, 0,-1], # move left
                [0, 0, -1], # move left
                [0, 0, 1], # move right
            ],
            "topleft_and_topright": [
                [0, 1, -1], # move left
                [0, -1, 1], # move right
                [0, 1, 1], # move right
                [0, -1,-1], # move left
                [0, 1, -1], # move left
                [0, -1, 1], # move right
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
        #self.agent.radius = 5.0
        self.agent.radius = 0.5
        colorlist = list(COLOR_NAMES)
        overall_types = []
        for obj_type in OBJ_TYPES:
            for color in colorlist:
                overall_types.append(obj_type + "_" + color)
        np.random.shuffle(overall_types)
        # [x,y,z,dir] 
        positions = {
            "front":       [self.size/2-4.5, 0, self.size/2,     0*np.pi/180], # front
            #"front-left":  [self.size/2-3.5, 0, self.size/2+3.5, 45*np.pi/180], # the front-left corner
            "left":        [self.size/2,     0, self.size/2+4.5, 90*np.pi/180], # left
            #"back-left":   [self.size/2+3.5, 0, self.size/2+3.5, 125*np.pi/180], # the back left corner
            #"back-right":  [self.size/2+3.5, 0, self.size/2-3.5, 225*np.pi/180], # the back right corner
            "right":       [self.size/2,     0, self.size/2-4.5, 270*np.pi/180], # right
            #"front-right": [self.size/2-3.5, 0, self.size/2-3.5, 315*np.pi/180], # the front right corner
            "back1":        [self.size/2+4.5, 0, self.size/2,     180*np.pi/180], # the back
            "back2":        [self.size/2+4.5, -10, self.size/2,   180*np.pi/180], # the back hidden
            "back3":        [self.size/2+4.5, -10, self.size/2,   180*np.pi/180], # the back hidden
            "back4":        [self.size/2+4.5, -10, self.size/2,   180*np.pi/180], # the back hidden
        }

        self._actions = []
        for p_idx, pos in enumerate(positions.values()):
            dir = pos[3]
            _type = overall_types[p_idx].split("_")[0]
            _color = overall_types[p_idx].split("_")[1]
            #_type = "box"
            #_color = "green"
            #_type = np.random.choice(OBJ_TYPES)
            if _type == "ball":
                self.place_entity(Ball(color=_color, size=0.9), pos=pos[:3], dir=dir)
            elif _type == "box":
                self.place_entity(Box(color=_color, size=0.9), pos=pos[:3], dir=dir)
            elif _type == "key":
                self.place_entity(Key(color=_color, size=0.7), pos=pos[:3], dir=dir)
            self._actions.append(self._action_set[self.np_random.choice(list(self._action_set.keys()))])
            #self._actions.append(self._action_set["circle_cw"])
        self._action_idxs = [0]*len(self._actions)
        self._back_idxs = [len(self._actions)-4, len(self._actions)-3, len(self._actions)-2, len(self._actions)-1]
        self._active_back_idx = 0

        self.place_agent(dir=225*np.pi/180, pos=[self.size/2, 0, self.size/2]) # start from 45 angle

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        if self.step_count == 54:
            self.entities[len(self._actions)-4].pos[1] -= 10 # back1
            self.entities[len(self._actions)-3].pos[1] += 10 # back2
            self._active_back_idx = 1
        if self.step_count == 72:
            self.entities[len(self._actions)-3].pos[1] -= 10 # back2
            self.entities[len(self._actions)-2].pos[1] += 10 # back3
            self._active_back_idx = 2
        if self.step_count == 90:
            self.entities[len(self._actions)-2].pos[1] -= 10 # back3
            self.entities[len(self._actions)-1].pos[1] += 10 # back4
            self._active_back_idx = 3
        for i in range(len(self._actions)):
            self.entities[i].pos[0] = self.entities[i].pos[0] + self._actions[i][self._action_idxs[i]][0] * np.cos(self.entities[i].dir) + self._actions[i][self._action_idxs[i]][2] * np.sin(self.entities[i].dir) # originally -sin, but used + in this env
            self.entities[i].pos[1] += self._actions[i][self._action_idxs[i]][1]
            self.entities[i].pos[2] = self.entities[i].pos[2] + self._actions[i][self._action_idxs[i]][0] * np.sin(self.entities[i].dir) + self._actions[i][self._action_idxs[i]][2] * np.cos(self.entities[i].dir)
            self._action_idxs[i] = (self._action_idxs[i] + 1) % len(self._actions[i])
        return obs, reward, termination, truncation, info

    def set_agent_dir(self, dir=0):
        self.agent.dir = dir
        
    def action_start(self):
        if self._action_idxs[0] == 0:
            return True
        else:
            return False
        
    def remove_objects(self):
        for i in range(len(self._actions)):
            self.entities[i].pos[1] -= 100

    def rollback_objects(self):
        for i in range(len(self._actions)):
            self.entities[i].pos[1] += 100

    def replace_back_object(self, back_idx=0):
        self.entities[self._back_idxs[self._active_back_idx]].pos[1] -= 10
        self.entities[self._back_idxs[back_idx]].pos[1] += 10
        self._active_back_idx = back_idx