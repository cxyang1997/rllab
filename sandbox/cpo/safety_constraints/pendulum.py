from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from sandbox.cpo.safety_constraints.base import *
import numpy as np

# x-position constraint. 
class PendulumSafetyConstraint(SafetyConstraint, Serializable):

    def __init__(self, max_value=1., lim=0.4, abs_lim=False, idx=0, CPO_version=None, **kwargs):
        self.lim = lim
        self.max_value = max_value
        self.abs_lim = abs_lim
        self.idx = idx
        self.verification_length = 10
        Serializable.quick_init(self, locals())
        super(PendulumSafetyConstraint,self).__init__(max_value, **kwargs)

    def evaluate(self, path, verifiable_safety_res=False):
        # measure the first 10 steps \in [-0.01, 0.01]
        if verifiable_safety_res:
            safety_res = path['observations'][:self.verification_length, self.idx] <= self.lim
        else:
            if path['observations'].shape[0] > self.verification_length:
                if self.CPO_version == 'CPO':
                    safety_res = path['observations'][:, self.idx] <= self.lim
                elif self.CPO_version == 'CPO_unsafety':
                    safety_res = path['observations'][:self.verification_length, self.idx] <= self.lim
                    padding = np.abs(path['observations'][self.verification_length:, self.idx]) >= 0 # all true
                    safety_res = np.concatenate((safety_res, ~padding)) # all the states beyond the verification step are set to false
                elif self.CPO_version == 'CPO_safety':
                    safety_res = path['observations'][:self.verification_length, self.idx] <= self.lim
                    padding = np.abs(path['observations'][self.verification_length:, self.idx]) >= 0 # all true
                    safety_res = np.concatenate((safety_res, ~padding)) # all the states beyond the verification step are set to false
            else:
                safety_res = path['observations'][:self.verification_length, self.idx] <= self.lim
        return safety_res

