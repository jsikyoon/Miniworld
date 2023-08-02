from gymnasium import spaces, utils

from miniworld.entity import Box
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS

import numpy as np

DEFAULT_MAX_FRAMES_PER_PHASE = {
    "first":      4,
    "second":  20,
    #"second":  80,
    "third":      16,
}

default_params = DEFAULT_PARAMS.no_random()
default_params.set("forward_step", 0.8)

class VisualMatch(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    ## Arguments

    ```python
    env = gym.make("MiniWorld-VisualMatch-v0")
    ```

    """

    def __init__(self, size=8, max_episode_steps=180, **kwargs):
        assert size >= 2
        self.size = size

        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        utils.EzPickle.__init__(
            self, size=size, max_episode_steps=max_episode_steps, **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        small_room = self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size) # small room with goal box
        large_room_size = 15
        large_room = self.add_rect_room(min_x=self.size+1, max_x=self.size+1+large_room_size, min_z=self.size+1, max_z=self.size+1+large_room_size) # large room with fruits

        self.goal_color = np.random.choice(["green", "blue"]) # randomly choose a color in the list
        self.green_box = self.place_entity(Box(color="green"))
        self.blue_box = self.place_entity(Box(color="blue"))
        if self.goal_color == "green":
            self.green_box.pos = (self.size/2, 0, self.size/2)
            self.blue_box.pos = (-1000, -1000, -1000)
        else:
            self.green_box.pos = (-1000, -1000, -1000)
            self.blue_box.pos = (self.size/2, 0, self.size/2)
       
        self.fruits = [] 
        for _ in range(15):
            self.fruits.append(self.place_entity(Box(color="yellow"), room=large_room, min_x=self.size+3, min_z=self.size+3))
            
        self.place_agent(dir=0, min_x=0, max_x=1, min_z=self.size/2-1, max_z=self.size/2)
        self._phase = "first"
        
    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        # move to the next phase
        if self._phase == "first" and self.step_count > DEFAULT_MAX_FRAMES_PER_PHASE[self._phase]:
            self.agent.pos = (self.size+2, 0, self.size+2)
            self.agent.dir = 0
            self._phase = "second"
        if self._phase == "second" and self.step_count > DEFAULT_MAX_FRAMES_PER_PHASE[self._phase] + DEFAULT_MAX_FRAMES_PER_PHASE["first"]:
            self.agent.pos = (1, 0, self.size/2)
            self.agent.dir = 0
            self.green_box.pos = np.array([self.size/2, 0, self.size/2-1.5])
            self.blue_box.pos = np.array([self.size/2, 0, self.size/2+1.5])
            self._phase = "third"
        obs = self.render_obs()

        if self._phase == "second":
            for fruit in self.fruits:
                if self.near(fruit):
                    self.fruits.remove(fruit)
                    reward += 0.1
                    fruit.pos -= 100 # remove the fruit from the world
                    
        if self._phase == "third":
            if self.near(self.green_box) and self.goal_color == "green":
                reward += 10
                termination = True
            elif self.near(self.blue_box) and self.goal_color == "blue":
                reward += 10
                termination = True
            elif self.near(self.green_box) and self.goal_color == "blue":
                termination = True
            elif self.near(self.blue_box) and self.goal_color == "green":
                termination = True
                    
        return obs, reward, termination, truncation, info