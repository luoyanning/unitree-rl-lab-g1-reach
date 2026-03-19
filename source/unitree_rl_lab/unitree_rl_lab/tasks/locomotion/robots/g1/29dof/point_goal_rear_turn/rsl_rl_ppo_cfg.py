from isaaclab.utils import configclass

from ..point_goal.rsl_rl_ppo_cfg import PointGoalPPORunnerCfg


@configclass
class PointGoalRearTurnPPORunnerCfg(PointGoalPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        # Fine-tune conservatively from the latest point-goal checkpoint.
        self.algorithm.learning_rate = 5.0e-6
