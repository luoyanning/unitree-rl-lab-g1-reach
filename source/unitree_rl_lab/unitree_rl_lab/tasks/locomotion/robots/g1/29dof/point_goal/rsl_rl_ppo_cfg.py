from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PointGoalPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        self.policy.init_noise_std = 0.4
        self.algorithm.learning_rate = 3.0e-4
        self.clip_actions = 1.0
