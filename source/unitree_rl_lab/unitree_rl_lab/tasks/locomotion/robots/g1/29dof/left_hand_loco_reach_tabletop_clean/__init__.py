import gymnasium as gym


gym.register(
    id="Unitree-G1-29dof-LeftHand-LocoReach-TableTopBalance-Clean-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_tabletop_clean_env_cfg:"
            "RobotLeftHandLocoReachTableTopBalanceCleanEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_tabletop_clean_env_cfg:"
            "RobotLeftHandLocoReachTableTopBalanceCleanPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.rsl_rl_ppo_cfg:LeftHandLocoReachTableTopBalanceCleanPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-LeftHand-LocoReach-TableTopPreTouch-Clean-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_tabletop_clean_env_cfg:"
            "RobotLeftHandLocoReachTableTopPreTouchCleanEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_tabletop_clean_env_cfg:"
            "RobotLeftHandLocoReachTableTopPreTouchCleanPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.rsl_rl_ppo_cfg:LeftHandLocoReachTableTopPreTouchCleanPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-LeftHand-LocoReach-TableTopTouch-Clean-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_tabletop_clean_env_cfg:"
            "RobotLeftHandLocoReachTableTopTouchCleanEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_tabletop_clean_env_cfg:"
            "RobotLeftHandLocoReachTableTopTouchCleanPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.rsl_rl_ppo_cfg:LeftHandLocoReachTableTopTouchCleanPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-LeftHand-LocoReach-TableTopMultiTouch-Clean-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_tabletop_clean_env_cfg:"
            "RobotLeftHandLocoReachTableTopMultiTouchCleanEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_tabletop_clean_env_cfg:"
            "RobotLeftHandLocoReachTableTopMultiTouchCleanPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.rsl_rl_ppo_cfg:LeftHandLocoReachTableTopMultiTouchCleanPPORunnerCfg"
        ),
    },
)
