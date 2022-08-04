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

class PendulumEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, g=9.81):
        self.max_speed = 1 # convert to 1
        self.max_torque = 1.0
        self.dt = 0.05
        self.g = g
        self.m = 0.25 # 1.0
        self.l = 2.0 # 1.0
        self.screen = None
        self.clock = None
        self.isopen = True

        self.screen_dim = 500
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        high = np.array([np.pi, self.max_speed], dtype=np.float32)
        # obs: theta, velocity
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.seed()

        _, self.nn_env = load_model("/home/cyang/vrl/vrl/exp/nn_models/pendulum")
        for p in self.nn_env.parameters():
            p.requires_grad = False
        # safe area: \thetat \on [-0.4, 0.4]
        
        self.cpo_log_path = f"/home/cyang/vrl/vrl/exp/pendulum/cpo/training.txt"

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u): # u is action
        th, thdot = self.state  # th := theta

        g = self.g 
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        # costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        # print(type(costs)) # np.float64
        reward = -abs(th)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        return np.array(self.state), reward, False, {}
    
    def nn_step(self, u):
        th, thdot = self.state
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        u = np.clip(u, -self.max_torque, self.max_torque)
        self.last_u = u  # for rendering
        # costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u[0]**2)
        # costs = costs[0]
        reward = -abs(th)
        dynamics_input = np.concatenate((self.state, u))
        dynamics_input = torch.from_numpy(dynamics_input).float()
        if torch.cuda.is_available():
            dynamics_input = dynamics_input.cuda()
        self.state = self.nn_dynamics(dynamics_input)
        self.state = self.state.detach().cpu().numpy() # convert from GPU to CPU
        return np.array(self.state), reward, False, {}
    
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

    def reset(
        self,
        *,
        # seed: Optional[int] = None,
        return_info: bool = False,
    ):
        # super().reset(seed=seed)
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}
    
    def random_action(self):
        return np.array(self.np_random.uniform(low=-1.0, high=1.0, size=(1,)))

    def render(self, mode="human"):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img, (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2)
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
    
