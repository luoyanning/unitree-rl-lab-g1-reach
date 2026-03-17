import gymnasium as gym


gym.register(
    id="Unitree-G1-29dof-LeftHand-LocoReach-AdapterHoldStay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_adapter_hold_stay_env_cfg:"
            "RobotLeftHandLocoReachAdapterHoldStayEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_adapter_hold_stay_env_cfg:"
            "RobotLeftHandLocoReachAdapterHoldStayPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:LeftHandLocoReachAdapterHoldStayPPORunnerCfg",
    },
)
