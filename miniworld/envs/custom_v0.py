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
        '''
        # x_pos = self.np_random.uniform(low=min_x + 2., high=max_x - 2.)
        # z_pos = self.np_random.uniform(low=min_z + 2., high=max_z - 2.)
        # pos = np.array([x_pos, 0, z_pos])
        x_pos = self.np_random.choice([self.np_random.uniform(low=min_x + 0.3, high=min_x + 0.5),
                                       self.np_random.uniform(low=max_x - 0.5, high=max_x - 0.3)])
        z_pos = self.np_random.choice([self.np_random.uniform(low=min_z + 0.3, high=min_z + 0.5),
                                       self.np_random.uniform(low=max_z - 0.5, high=max_z - 0.3)])
        pos = np.array([x_pos, 0, z_pos])
        '''
        '''  # loop test position spawn
        pos_dir = np.random.permutation([[np.array([-4, 0, -4]), 0.],
                                         [np.array([4, 0, -4]), math.pi],
                                         [np.array([-4, 0, 4]), 0.],
                                         [np.array([4, 0, 4]), math.pi],
                                         ])[0]
        pos, dir = pos_dir
        self.place_agent(room=room0,
                         pos=pos,
                         dir=dir,)
        '''
        
        # generate spawn poses
        x_lin = np.linspace(0, int(self.max_x - self.min_x)-1, int(self.max_x - self.min_x)) - 4.5
        y_lin = np.linspace(0, int(self.max_z - self.min_z)-1, int(self.max_z - self.min_z)) - 4.5
        X_lin, Y_lin = np.meshgrid(x_lin, y_lin)
        spawn_poss = np.concatenate([X_lin.flatten()[..., None],
                                     np.zeros_like(X_lin.flatten())[..., None],
                                     Y_lin.flatten()[..., None]], axis=1)
        
        self.spawn_nowall, self.spawn_wall, self.spawn = [], [], []
        for spwan_pos in spawn_poss:
            if spwan_pos[0] > 4.0 or spwan_pos[0] < -4.0 or spwan_pos[2] > 4.0 or spwan_pos[2] < -4.0:
                self.spawn_wall.append(spwan_pos)
            else:
                self.spawn_nowall.append(spwan_pos)
            self.spawn.append(spwan_pos)
        
        # whether to spawn pose close to wall or not
        control_wall = True
        close_to_wall = False
        if control_wall:
            if close_to_wall:
                pos = self.np_random.choice(self.spawn_wall)
            else:
                pos = self.np_random.choice(self.spawn_nowall)
        else:
            pos = self.np_random.choice(self.spawn)
        
        
        '''
        # select direction
        if pos[0] < 0 and pos[2] < 0:
            dir = np.random.uniform(-np.pi * (1/2), -np.pi * (1/3))
        elif pos[0] > 0 and pos[2] < 0:
            dir = np.random.uniform(-np.pi, -np.pi * (5/6))
        elif pos[0] > 0 and pos[2] > 0:
            dir = np.random.uniform(np.pi * (1/2), np.pi * (2/3))
        else:
            dir = np.random.uniform(0, np.pi * (1/6))
        '''
        dir = 0.0
        self.place_agent(room=room0,
                         pos=pos,
                         dir=dir,)
        

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        return obs, reward, termination, truncation, info