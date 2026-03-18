import gymnasium as gym


gym.register(
    id="Unitree-G1-29dof-LeftHand-LocoReach-AdapterAcquireTightStay-SoftDwellHardStart-HandStill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_adapter_acquire_tight_stay_soft_dwell_hardstart_handstill_env_cfg:"
            "RobotLeftHandLocoReachAdapterAcquireTightStaySoftDwellHardStartHandStillEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_adapter_acquire_tight_stay_soft_dwell_hardstart_handstill_env_cfg:"
            "RobotLeftHandLocoReachAdapterAcquireTightStaySoftDwellHardStartHandStillPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.rsl_rl_ppo_cfg:LeftHandLocoReachAdapterAcquireTightStaySoftDwellHardStartHandStillPPORunnerCfg"
        ),
    },
)
