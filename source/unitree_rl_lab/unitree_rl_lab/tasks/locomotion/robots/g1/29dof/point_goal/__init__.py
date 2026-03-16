import gymnasium as gym


gym.register(
    id="Unitree-G1-29dof-PointGoal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.point_goal_env_cfg:RobotPointGoalEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.point_goal_env_cfg:RobotPointGoalPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:PointGoalPPORunnerCfg",
    },
)
