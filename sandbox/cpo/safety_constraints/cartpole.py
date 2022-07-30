from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from sandbox.cpo.safety_constraints.base import *
import numpy as np

# x-position constraint. 
class CartpoleSafetyConstraint(SafetyConstraint, Serializable):

    def __init__(self, max_value=1., lim=0.01, abs_lim=False, idx=0, **kwargs):
        self.lim = lim
        self.max_value = max_value
        self.abs_lim = abs_lim
        self.idx = idx
        Serializable.quick_init(self, locals())
        super(CartpoleSafetyConstraint,self).__init__(max_value, **kwargs)

    def evaluate(self, path):
        #return np.zeros_like(path['rewards'])
        # return np.abs(path['observations'][:10, self.idx]) <= self.lim
        return np.abs(path['observations'][:, self.idx]) <= self.lim

        # if not(self.abs_lim):
        #     return path['observations'][:,self.idx] >= self.lim
        # else:
        #     return np.abs(path['observations'][:,self.idx]) >= self.lim
