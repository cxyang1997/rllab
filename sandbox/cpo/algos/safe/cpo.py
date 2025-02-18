from sandbox.cpo.algos.safe.policy_gradient_safe import PolicyGradientSafe
from sandbox.cpo.optimizers.conjugate_constraint_optimizer import ConjugateConstraintOptimizer
from rllab.core.serializable import Serializable
from sandbox.cpo.algos.safe.utils import ini_training_log


class CPO(PolicyGradientSafe, Serializable):
    """
    Constrained Policy Optimization
    """

    def __init__(
            self,
            env=None,
            optimizer=None,
            optimizer_args=None,
            safety_constraint=None,
            safety_tradeoff=False,
            learn_safety_tradeoff_coeff=False,
            max_path_length=None,
            CPO_version=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateConstraintOptimizer(**optimizer_args)

        if 'safety_constrained_optimizer' in kwargs.keys():
            kwargs.pop('safety_constrained_optimizer')

        super(CPO, self).__init__(env=env,
                                    optimizer=optimizer, 
                                   safety_constrained_optimizer=True,
                                   safety_constraint=safety_constraint,
                                   safety_tradeoff=safety_tradeoff,
                                   learn_safety_tradeoff_coeff=learn_safety_tradeoff_coeff,
                                   max_path_length=max_path_length,
                                   **kwargs)
        self.cpo_training_log_path = f"{env.cpo_log_path}/{CPO_version}/training.txt"
        self.max_path_length = max_path_length
        self.training_f = ini_training_log(self.cpo_training_log_path)

