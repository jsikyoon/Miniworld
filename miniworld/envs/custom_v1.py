import math
import numpy as np
from gymnasium import spaces, utils

from miniworld.entity import Box, MeshEnt
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS

# Parameters for larger movement steps, fast stepping
default_params = DEFAULT_PARAMS.no_random()
default_params.set("forward_step", 0.5)  # default value 0.15
# default_params.set("turn_step", 60)  # default value 15
# default_params.set("cam_fov_y", 170)  # default value 60
obs_width=84
obs_height=84
window_width=840
window_height=840


class CustomV1(MiniWorldEnv, utils.EzPickle):
    """
    ## Description
    Single room for learning position (No touch wall)
    ## Action Space
    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |
    | 3   | move back                   |
    ## Observation Space
    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing the view the agents sees.
    """

    def __init__(self, **kwargs):
        MiniWorldEnv.__init__(self, max_episode_steps=1000, 
                              params=default_params,
                              obs_width=obs_width,
                              obs_height=obs_height,
                              window_width=window_width,
                              window_height=window_height,
                              **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_back + 1)

    def _gen_world(self):
        
        # Maze Generation
        nr = 5
        nc = 5
        gap = 2
        rooms = [[None for _ in range(nr)] for _ in range(nc)]
  
        for j in range(nc):            
            for i in range(nr):
                
                min_x = 10*i + gap*i
                max_x = 10*(i + 1) + gap*i
                min_z = 10*j + gap*j
                max_z = 10*(j + 1) + gap*j
                
                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex="brick_wall",
                    floor_tex="asphalt",
                    no_ceiling=True,
                )
                
                rooms[j][i] = room
                
                if j == 0:
                    if i > 0:
                        self.connect_rooms(rooms[j][i-1], rooms[j][i], 
                                           min_z=3+10*j+gap*j, 
                                           max_z=7+10*j+gap*j)
                else:
                    if i == 0:
                        self.connect_rooms(rooms[j-1][i], rooms[j][i], 
                                           min_x=3+10*i+gap*i, 
                                           max_x=7+10*i+gap*i)

                    else:
                        self.connect_rooms(rooms[j-1][i], rooms[j][i], 
                                           min_x=3+10*i+gap*i, 
                                           max_x=7+10*i+gap*i)
                        self.connect_rooms(rooms[j][i-1], rooms[j][i], 
                                           min_z=3+10*j+gap*j, 
                                           max_z=7+10*j+gap*j)
            
        self.place_agent(room=rooms[0][0],
                         pos=np.array([5., 0., 5.]),
                         dir=0)

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        return obs, reward, termination, truncation, info