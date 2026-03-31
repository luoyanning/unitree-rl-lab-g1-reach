from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class LeftHandLocoReachTableTopTouchPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        self.policy.init_noise_std = 0.15
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.desired_kl = 0.003
        self.algorithm.entropy_coef = 0.002
        self.save_interval = 50
