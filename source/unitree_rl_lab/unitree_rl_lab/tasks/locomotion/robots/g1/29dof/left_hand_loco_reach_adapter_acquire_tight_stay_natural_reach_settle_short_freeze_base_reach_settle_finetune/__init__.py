import gymnasium as gym


gym.register(
    id="Unitree-G1-29dof-LeftHand-LocoReach-AdapterAcquireTightStay-NaturalReachSettleShort-FreezeBaseReach-SettleFinetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_adapter_acquire_tight_stay_natural_reach_settle_short_freeze_base_reach_settle_finetune_env_cfg:"
            "RobotLeftHandLocoReachAdapterAcquireTightStayNaturalReachSettleShortFreezeBaseReachSettleFinetuneEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.left_hand_loco_reach_adapter_acquire_tight_stay_natural_reach_settle_short_freeze_base_reach_settle_finetune_env_cfg:"
            "RobotLeftHandLocoReachAdapterAcquireTightStayNaturalReachSettleShortFreezeBaseReachSettleFinetunePlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.rsl_rl_ppo_cfg:"
            "LeftHandLocoReachAdapterAcquireTightStayNaturalReachSettleShortFreezeBaseReachSettleFinetunePPORunnerCfg"
        ),
    },
)
