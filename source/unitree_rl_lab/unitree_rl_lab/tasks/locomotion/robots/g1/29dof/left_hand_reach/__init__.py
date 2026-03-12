import gymnasium as gym


gym.register(
    id="Unitree-G1-29dof-LeftHand-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.left_hand_reach_env_cfg:RobotLeftHandReachEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.left_hand_reach_env_cfg:RobotLeftHandReachPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
