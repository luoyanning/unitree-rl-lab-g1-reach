import gymnasium as gym

from .left_hand_reach_env_cfg import G1LeftHandReachEnvCfg, G1LeftHandReachEnvCfg_PLAY

gym.register(
    id="Unitree-G1-29dof-LeftHand-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": G1LeftHandReachEnvCfg,
        # 这里先直接复用你当前 G1 29DOF velocity 的 PPO 配置
        # 后面再单独拷一份 reach 专用 cfg
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.agents.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-LeftHand-Reach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": G1LeftHandReachEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.agents.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)