import math
import numpy as np
from gymnasium import spaces, utils

from miniworld.entity import Box, MeshEnt
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS

# Parameters for larger movement steps, fast stepping
default_params = DEFAULT_PARAMS.no_random()
default_params.set("forward_step", 0.5)
# default_params.set("turn_step", 45)
obs_width=84
obs_height=84
window_width=840
window_height=840


class CustomV0(MiniWorldEnv, utils.EzPickle):
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
        
        min_x, max_x, min_z, max_z = -5, 5, -5, 5
        
        # Top
        room0 = self.add_rect_room(
            min_x=min_x,
            max_x=max_x,
            min_z=min_z,
            max_z=max_z,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )

        # x: [-3, 3]
        # z: [-3, 3]
        #x_pos = self.np_random.uniform(low=min_x + 2., high=max_x - 2.)
        #z_pos = self.np_random.uniform(low=min_z + 2., high=max_z - 2.)
        #pos = np.array([x_pos, 0, z_pos])
        
        # x: {-4.5 -3.5 -2.5 -1.5 -0.5  0.5  1.5  2.5  3.5  4.5}
        # z: {-4.5 -3.5 -2.5 -1.5 -0.5  0.5  1.5  2.5  3.5  4.5}
        x_lin = np.linspace(0, int(max_x - min_x)-1, int(max_x - min_x)) - 4.5
        y_lin = np.linspace(0, int(max_z - min_z)-1, int(max_z - min_z)) - 4.5
        X_lin, Y_lin = np.meshgrid(x_lin, y_lin)
        spawn_poss = np.concatenate([X_lin.flatten()[..., None],
                                     np.zeros_like(X_lin.flatten())[..., None],
                                     Y_lin.flatten()[..., None]], axis=1)
        spawn_nowall = []
        for spwan_pos in spawn_poss:
            if not (spwan_pos[0] == 4.5 or spwan_pos[0] == -4.5 or spwan_pos[2] == 4.5 or spwan_pos[2] == -4.5):
                spawn_nowall.append(spwan_pos)
        pos = self.np_random.choice(spawn_nowall)
        
        self.place_agent(room=room0, pos=pos)

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        return obs, reward, termination, truncation, info