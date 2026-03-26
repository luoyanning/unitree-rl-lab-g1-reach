from isaaclab.utils import configclass

from ..left_hand_loco_reach_adapter_acquire_tight_stay_natural_reach_settle_short_freeze_base_reach.rsl_rl_ppo_cfg import (
    LeftHandLocoReachAdapterAcquireTightStayNaturalReachSettleShortFreezeBaseReachPPORunnerCfg,
)


@configclass
class LeftHandLocoReachAdapterAcquireTightStayNaturalReachSettleShortFreezeBaseReachSettleFinetunePPORunnerCfg(
    LeftHandLocoReachAdapterAcquireTightStayNaturalReachSettleShortFreezeBaseReachPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.35
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.desired_kl = 0.005
        self.algorithm.entropy_coef = 0.005
        self.save_interval = 50
