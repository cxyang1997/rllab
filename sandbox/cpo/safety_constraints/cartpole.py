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
        self.verification_length = 10
        Serializable.quick_init(self, locals())
        super(CartpoleSafetyConstraint,self).__init__(max_value, **kwargs)

    def evaluate(self, path):
        # measure the first 10 steps \in [-0.01, 0.01]
        # return np.abs(path['observations'][:10, self.idx]) <= self.lim
        if path['observations'].shape[0] > self.verification_length :
            safety_res = np.abs(path['observations'][:self.verification_length, self.idx]) <= self.lim
            padding = np.abs(path['observations'][self.verification_length:, self.idx]) >= 0 # all true
            # print(f"safety: {safety_res.shape}; padding: {padding.shape}")
            # print(f"safety: {safety_res}; padding: {padding}")
            safety_res = np.concatenate((safety_res, ~padding)) # all the states beyond the verification step are set to false
            # print(f"final res: {safety_res.shape}")
            # print(f"final res: {safety_res}")
        else:
            safety_res = np.abs(path['observations'][:self.verification_length, self.idx]) <= self.lim
            # print(f"final res: {safety_res.shape}")
            # print(f"final res: {safety_res}")
        return safety_res

