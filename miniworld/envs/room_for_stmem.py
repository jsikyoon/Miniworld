import math

from gymnasium import utils

from miniworld.entity import COLOR_NAMES, Ball, Box, Key
from miniworld.miniworld import MiniWorldEnv

import numpy as np

obs_width=84
obs_height=84
window_width=840
window_height=840

class RoomSTMEM(MiniWorldEnv, utils.EzPickle):
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

        #for _ in range(10):
        #    #self.place_entity(
        #    #    Box(color=colorlist[self.np_random.choice(len(colorlist))], size=0.9)
        #    #)
        #    self.place_entity(
        #        Ball(color=colorlist[self.np_random.choice(len(colorlist))], size=0.9)
        #    )
        #self.place_entity(Key(color=colorlist[self.np_random.choice(len(colorlist))]))
        
        self.place_agent(dir=180*np.pi/180, pos=[self.size/2, 0, self.size/2])

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        return obs, reward, termination, truncation, info
