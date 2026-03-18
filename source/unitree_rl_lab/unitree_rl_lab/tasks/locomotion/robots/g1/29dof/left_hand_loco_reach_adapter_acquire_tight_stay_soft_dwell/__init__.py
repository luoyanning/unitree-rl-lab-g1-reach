import gymnasium as gym


gym.register(
    id="Unitree-G1-29dof-LeftHand-LocoReach-AdapterAcquireTightStay-SoftDwell-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_adapter_acquire_tight_stay_soft_dwell_env_cfg:"
            "RobotLeftHandLocoReachAdapterAcquireTightStaySoftDwellEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_adapter_acquire_tight_stay_soft_dwell_env_cfg:"
            "RobotLeftHandLocoReachAdapterAcquireTightStaySoftDwellPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.rsl_rl_ppo_cfg:LeftHandLocoReachAdapterAcquireTightStaySoftDwellPPORunnerCfg"
        ),
    },
)
