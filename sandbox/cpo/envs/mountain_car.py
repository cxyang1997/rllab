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

class MountainCarEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, goal_velocity=0):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.7
        # self.max_speed = 0.007
        self.max_speed = 7.0
        self.goal_position = 0.6
        self.goal_velocity = goal_velocity
        # self.power = 0.001
        self.power = 1.0
        # self.gravity = 0.0025
        self.gravity = 2.5
        self.dt = 0.05

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.screen = None
        self.clock = None
        self.isopen = True

        act_high = np.array((1,), dtype=np.float32)
        self.action_space = spaces.Box(-act_high, act_high, dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        _, self.nn_env = load_model("/home/cyang/vrl/vrl/exp/nn_models/mountain_car")
        for p in self.nn_env.parameters():
            p.requires_grad = False
        # safe area: \position \on [-pi/3, +oo]
        
        self.cpo_log_path = f"/home/cyang/vrl/vrl/exp/mountain_car/cpo/training.txt"

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action): # u is action
        action = action.squeeze() # The action here is [-1, 1]
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action, self.min_action), self.max_action)
        dt = self.dt
        # force = action

        position += velocity * dt
        # velocity += (force * self.power - self.gravity * math.cos(3 * position))
        velocity += dt * (force * self.power - self.gravity * math.cos(3 * position))
        self.state = np.clip(
            np.array([position, velocity]),
            self.low, 
            self.high
        )
        # if velocity > self.max_speed:
        #     velocity = self.max_speed
        # if velocity < -self.max_speed:
        #     velocity = -self.max_speed
        # position += dt * velocity
        # if position > self.max_position:
        #     position = self.max_position
        # if position < self.min_position:
        #     position = self.min_position
        # if position == self.min_position and velocity < 0:
        #     velocity = 0

        # Convert a possible numpy bool to a Python bool.
        done = bool(position >= self.goal_position)
        # done = bool(position >= self.goal_position)

        # reward = 0
        if done:
            reward = 1000.0
        # sparse reward
        # reward -= math.pow(action, 2) * 0.1
        reward = -1
        # dense reward
        # reward = position
        # reward = position - 0.6

        self.state = np.array([position, velocity], dtype=np.float32)
        return np.array(self.state, dtype=np.float32), reward, done, {}
    
    def nn_step(self, action):
        dynamics_input = np.concatenate((self.state, action))
        dynamics_input = torch.from_numpy(dynamics_input).float()
        if torch.cuda.is_available():
            dynamics_input = dynamics_input.cuda()
        self.state = self.nn_dynamics(dynamics_input)
        self.state = self.state.detach().cpu() # convert from GPU to CPU
        position, velocity = self.state

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = 0
        if done:
            reward = 1000.0
        # reward -= math.pow(action, 2) * 0.1
        reward = -1

        self.state = np.array([position, velocity], dtype=np.float32)
        return np.array(self.state, dtype=np.float32), reward, done, {}
    
    def run_dynamics(self, action, state, metrics):
        if metrics == 'symbolic':
            symbolic_state = self.symbolic_dynamics(action, state)
        if metrics == 'nn':
            symbolic_state = self.nn_dynamics(action, state)
        return symbolic_state
    
    # TODO: linear approximation of pendulum
    def symbolic_dynamics(self, action, state):
        # TODO: do not add clip option for now
        # approximate the dynamics 
        # f(thdot) = f(0) + f'(0)*thdot
        # f(thdot) = (3/(m * l**2) * g) * dt
        # f(theta) = f(0) + f'(0)*theta
        # f(theta) = (thdot * dt) + (3 / (m * l^2) * u * (dt)**2) + theta * (1 + 3 * g / (2 * l) * dt**2 + 3 / (m * l**2) * u * dt**2)
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        theta = state.select_from_index(1, index0)
        thdot = state.select_from_index(1, index1)
        new_thdot = thdot.add((3 / (m * l**2) * g) * dt)
        new_theta = theta.mul(1 + 3 * g / (2 * l) * (dt)**2).add(theta.mul(action).mul(3 / (m * l**2) * (dt)**2))
        symbolic_state = new_theta.concatenate(new_thdot)

        return symbolic_state

    # The NN approximation of cartpole
    def nn_dynamics(self, action, state):
        # only for verification
        symbolic_state = self.nn_env(state.concatenate(action))
        return symbolic_state

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.55, high=-0.45), 0])
        self.steps_beyond_done = None
        return np.array(self.state)
    
    def random_action(self):
        return np.array(self.np_random.uniform(low=-1.0, high=1.0, size=(1,)))

    def render(self, mode="human"):
        pass
        # TODO: add the rendering for mountain_car

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    
