from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from sandbox.cpo.safety_constraints.base import *
import numpy as np

# x-position constraint. 
class CartpoleSafetyConstraint(SafetyConstraint, Serializable):

    def __init__(self, max_value=1., lim=0.05, abs_lim=False, idx=0, CPO_version=None, **kwargs):
        self.lim = lim
        self.max_value = max_value
        self.abs_lim = abs_lim
        self.idx = idx
        self.verification_length = 10
        self.CPO_version = CPO_version # ['CPO', 'CPO_unsafety', 'CPO_safety', 'CPO_no_safety']
        Serializable.quick_init(self, locals())
        super(CartpoleSafetyConstraint,self).__init__(max_value, **kwargs)

    def evaluate(self, path, verifiable_safety_res=False): # calculate the safety cost
        # measure the first 10 steps \in [-0.05, 0.05]
        # this is used as a cost signal
        # cost: unsafe; no cost: safe
        if verifiable_safety_res: # The safety metrics in VRL
            safety_res = np.abs(path['observations'][:self.verification_length, self.idx]) <= self.lim
        elif self.CPO_version == 'CPO_no_safety':
            safety_res = np.abs(path['observations'][:, self.idx]) < 0 # all cost == 0
        else:
            if path['observations'].shape[0] > self.verification_length:
                if self.CPO_version == 'CPO':
                    safety_res = np.abs(path['observations'][:, self.idx]) > self.lim
                elif self.CPO_version == 'CPO_unsafety':
                    safety_res = np.abs(path['observations'][:self.verification_length, self.idx]) > self.lim
                    padding = np.abs(path['observations'][self.verification_length:, self.idx]) >= 0 # all true
                    safety_res = np.concatenate((safety_res, padding)) # all the states beyond the verification step are set to having cost
                elif self.CPO_version == 'CPO_safety':
                    safety_res = np.abs(path['observations'][:self.verification_length, self.idx]) > self.lim
                    padding = np.abs(path['observations'][self.verification_length:, self.idx]) >= 0 # all true
                    safety_res = np.concatenate((safety_res, ~padding)) # all the states beyond the verification step are set to cost==0
            else:
                safety_res = np.abs(path['observations'][:self.verification_length, self.idx]) > self.lim
        return safety_res

