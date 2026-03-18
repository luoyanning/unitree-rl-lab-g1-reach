import gymnasium as gym


gym.register(
    id="Unitree-G1-29dof-LeftHand-LocoReach-AdapterAcquireTightStay-SoftDwellHardStart-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_adapter_acquire_tight_stay_soft_dwell_hardstart_env_cfg:"
            "RobotLeftHandLocoReachAdapterAcquireTightStaySoftDwellHardStartEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_adapter_acquire_tight_stay_soft_dwell_hardstart_env_cfg:"
            "RobotLeftHandLocoReachAdapterAcquireTightStaySoftDwellHardStartPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.rsl_rl_ppo_cfg:LeftHandLocoReachAdapterAcquireTightStaySoftDwellHardStartPPORunnerCfg"
        ),
    },
)
