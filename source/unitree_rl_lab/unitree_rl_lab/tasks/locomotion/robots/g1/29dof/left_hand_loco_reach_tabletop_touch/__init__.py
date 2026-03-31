import gymnasium as gym


gym.register(
    id="Unitree-G1-29dof-LeftHand-LocoReach-TableTopTouch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.left_hand_loco_reach_tabletop_touch_env_cfg:RobotLeftHandLocoReachTableTopTouchEnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_tabletop_touch_env_cfg:"
            "RobotLeftHandLocoReachTableTopTouchPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:LeftHandLocoReachTableTopTouchPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-LeftHand-LocoReach-TableTopTouch-StandTouch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_tabletop_touch_env_cfg:"
            "RobotLeftHandLocoReachTableTopTouchStandTouchEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_tabletop_touch_env_cfg:"
            "RobotLeftHandLocoReachTableTopTouchStandTouchPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:LeftHandLocoReachTableTopTouchPPORunnerCfg",
    },
)
