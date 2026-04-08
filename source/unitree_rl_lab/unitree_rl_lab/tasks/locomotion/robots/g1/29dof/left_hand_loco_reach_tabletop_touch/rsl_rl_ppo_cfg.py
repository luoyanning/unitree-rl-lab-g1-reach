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


@configclass
class LeftHandLocoReachTableTopTouchV1PPORunnerCfg(BasePPORunnerCfg):
    """
    PPO config for V1 / V2 tabletop touch (clean from-scratch training).

    Tuning rationale:
      - Higher LR (1e-4) than the refined tabletop variants: task is simpler
        and we want fast early learning.
      - Moderate entropy (1.5e-3): enough exploration to discover the touch but
        not so much that the policy never commits to a reach.
      - init_noise_std=0.10: loose enough to explore arm configurations.
      - desired_kl=0.005: somewhat more aggressive updates acceptable for the
        early simple task.
    """
    def __post_init__(self):
        self.policy.init_noise_std = 0.10
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.desired_kl = 0.005
        self.algorithm.entropy_coef = 1.5e-3
        self.save_interval = 50
