from os import path

import gym
# from gym import Env
from .gym_env_base import Env
import numpy as np
import torch
# from gym import logger, spaces
from gym.utils import seeding
from gym.error import DependencyNotInstalled

from rllab import spaces
from rllab.misc import logger
# from rllab.envs.base import Env, Step
from rllab.envs.env_spec import EnvSpec
from cached_property import cached_property
from rllab.misc.overrides import overrides

import vrl.domain
from vrl.util.common import (
    var,
)
from vrl.env.nn_models.nn_as_env import (
    load_model,
)

index0 = torch.tensor(0)
index1 = torch.tensor(1)
index2 = torch.tensor(2)
index3 = torch.tensor(3)

if torch.cuda.is_available():
    index0 = index0.cuda()
    index1 = index1.cuda()
    index2 = index2.cuda()
    index3 = index3.cuda()

class AccEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self):

        self.low = np.array([-10, -10], dtype=np.float32)
        self.high = np.array([10, 10], dtype=np.float32)

        act_high = np.array((2,), dtype=np.float32)
        self.action_space = spaces.Box(-act_high, act_high, dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.rng = np.random.default_rng()

        self.seed()
        self.state = None

        self.steps_beyond_done = None
        _, self.nn_env = load_model("/home/cyang/vrl/vrl/exp/nn_models/acc")
        if self.nn_env is not None:
            for p in self.nn_env.parameters():
                p.requires_grad = False
        
        self.cpo_log_path = f"/home/cyang/vrl/vrl/exp/acc/cpo/training.txt"

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = action.squeeze() # The action here is [-1, 1]
        position = self.state[0]
        velocity = self.state[1]
        x = position + 0.1 * velocity
        v = velocity + 0.1 * action + self.rng.normal(loc=0, scale=0.5)
        self.state = np.clip(
            np.asarray([x, v]),
            self.observation_space.low,
            self.observation_space.high,
        )
        reward = 2.0 - x if x < 0 else -10
        
        return np.array(self.state, dtype=np.float32), reward, False, {}
    
    def nn_step(self, action):
        dynamics_input = np.concatenate((self.state, action))
        dynamics_input = torch.from_numpy(dynamics_input).float()
        if torch.cuda.is_available():
            dynamics_input = dynamics_input.cuda()
        self.state = self.nn_dynamics(dynamics_input)
        self.state = self.state.detach().cpu() # convert from GPU to CPU
        position, velocity = self.state

        reward = 2.0 - position if position < 0 else -10

        self.state = np.array([position, velocity], dtype=np.float32)
        return np.array(self.state, dtype=np.float32), reward, False, {}
    
    def run_dynamics(self, action, state, metrics):
        if metrics == 'symbolic':
            symbolic_state = self.symbolic_dynamics(action, state)
        if metrics == 'nn':
            symbolic_state = self.nn_dynamics(action, state)
        return symbolic_state
    
    # The linear approximation of mountain
    def symbolic_dynamics(self, action, state):
        position = state.select_from_index(1, index0)
        velocity = state.select_from_index(1, index1)

        position = position.add(velocity.mul(var(0.1)))
        position = position.clamp(-10, 10)
        velocity = velocity.add(action.mul(var(0.1))).widen(width=0.5) # for the random rng
        velocity = velocity.clamp(-10, 10)

        symbolic_state = position.concatenate(velocity)
        return symbolic_state

    # The NN approximation of cartpole
    def nn_dynamics(self, action, state):
        # only for verification
        symbolic_state = self.nn_env(state.concatenate(action))
        return symbolic_state

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-1.1, high=-0.9), \
            self.np_random.uniform(low=-0.1, high=0.1)])
        self.steps_beyond_done = None
        return np.array(self.state)
    
    def random_action(self):
        return np.array(self.np_random.uniform(low=-2.0, high=2.0, size=(1,)))

    def render(self, mode="human"):
        pass
        # TODO: add the rendering for mountain_car

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    
