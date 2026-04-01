from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class LeftHandLocoReachTableTopTouchPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        self.policy.init_noise_std = 0.10
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.desired_kl = 0.002
        self.algorithm.entropy_coef = 0.0015
        self.save_interval = 50


@configclass
class LeftHandLocoReachTableTopTouchScratchPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        self.policy.init_noise_std = 0.08
        self.algorithm.learning_rate = 8.0e-5
        self.algorithm.desired_kl = 0.004
        self.algorithm.entropy_coef = 0.001
        self.save_interval = 50
