import gymnasium as gym


gym.register(
    id="Unitree-G1-29dof-HomieLowerBody-v0",
    entry_point=f"{__name__}.homie_lower_body_env:G1HomieLowerBodyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.homie_lower_body_env:G1HomieLowerBodyEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.homie_lower_body_env:G1HomieLowerBodyPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:HomieLowerBodyPPORunnerCfg",
    },
)
