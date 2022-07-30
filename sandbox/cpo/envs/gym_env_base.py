from gym import Env
import numpy as np

from cached_property import cached_property
from rllab.misc.overrides import overrides
from rllab.envs.env_spec import EnvSpec

class Env(Env):
    def action_dim(self):
        return np.prod(self.action_space.low.shape)
    
    @cached_property
    def spec(self):
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
    
    @overrides
    def log_diagnostics(self, path):
        # no log for now
        pass

    def terminate(self):
        pass
    
    def horizon(self):
        """
        Horizon of the environment, if it has one
        """
        raise NotImplementedError
    
    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass




    