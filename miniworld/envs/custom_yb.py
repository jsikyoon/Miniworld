import math
import numpy as np
from gymnasium import spaces, utils

from miniworld.entity import Box, MeshEnt
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS

# Parameters for larger movement steps, fast stepping
default_params = DEFAULT_PARAMS.no_random()
default_params.set("forward_step", 0.20, 0.10, 0.30)  # default, min, max
default_params.set("turn_step", 10, 5, 15)  # default, min, max
obs_width=128
obs_height=128
window_width=640
window_height=640
domain_rand=True


class CustomYB(MiniWorldEnv, utils.EzPickle):
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
                              domain_rand=domain_rand,
                              **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_back + 1)

    def _gen_world(self):
        
         # map specifications
        self.min_x, self.max_x, self.min_z, self.max_z = -5, 5, -5, 5
        
        # Top
        room0 = self.add_rect_room(
            min_x=self.min_x,
            max_x=self.max_x,
            min_z=self.min_z,
            max_z=self.max_z,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )
        
        # generate spawn poses
        x_lin = np.linspace(0, int(self.max_x - self.min_x)-1, int(self.max_x - self.min_x)) - 4.5
        y_lin = np.linspace(0, int(self.max_z - self.min_z)-1, int(self.max_z - self.min_z)) - 4.5
        X_lin, Y_lin = np.meshgrid(x_lin, y_lin)
        spawn_poss = np.concatenate([X_lin.flatten()[..., None],
                                     np.zeros_like(X_lin.flatten())[..., None],
                                     Y_lin.flatten()[..., None]], axis=1)
        
        self.spawn_nowall, self.spawn_wall = [], []
        for spwan_pos in spawn_poss:
            if spwan_pos[0] == 4.5 or spwan_pos[0] == -4.5 or spwan_pos[2] == 4.5 or spwan_pos[2] == -4.5:
                self.spawn_wall.append(spwan_pos)
            else:
                self.spawn_nowall.append(spwan_pos)
        
        # whether to spawn pose close to wall or not
        wall = False
        if wall:
            pos = self.np_random.choice(self.spawn_wall)
        else:
            pos = self.np_random.choice(self.spawn_nowall)
            
        self.place_agent(room=room0,
                         pos=pos,)
        

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        return obs, reward, termination, truncation, info