from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PointGoalPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        self.policy.init_noise_std = 0.05
        self.algorithm.learning_rate = 1.0e-5
        self.algorithm.entropy_coef = 0.0
        self.clip_actions = 1.0
