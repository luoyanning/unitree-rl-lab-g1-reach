from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class LeftHandLocoReachTableTopBalanceCleanPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        self.policy.init_noise_std = 0.08
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.desired_kl = 0.004
        self.algorithm.entropy_coef = 0.0015
        self.save_interval = 50


@configclass
class LeftHandLocoReachTableTopPreTouchCleanPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        self.policy.init_noise_std = 0.03
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.desired_kl = 0.002
        self.algorithm.entropy_coef = 0.0005
        self.save_interval = 25


@configclass
class LeftHandLocoReachTableTopTouchCleanPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        self.policy.init_noise_std = 0.03
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.desired_kl = 0.002
        self.algorithm.entropy_coef = 0.0005
        self.save_interval = 25


@configclass
class LeftHandLocoReachTableTopTouchSpreadCleanPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        self.policy.init_noise_std = 0.04
        self.algorithm.learning_rate = 4.0e-5
        self.algorithm.desired_kl = 0.002
        self.algorithm.entropy_coef = 0.0005
        self.save_interval = 25


@configclass
class LeftHandLocoReachTableTopMultiTouchCleanPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        self.policy.init_noise_std = 0.06
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.desired_kl = 0.003
        self.algorithm.entropy_coef = 0.0008
        self.save_interval = 50
