from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

try:
    from isaaclab_rl.rsl_rl import RslRlSymmetryCfg
except ImportError:
    RslRlSymmetryCfg = None

from .homie_lower_body_symmetry import maybe_get_symmetry_cfg_kwargs


@configclass
class HomieLowerBodyPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        self.policy.init_noise_std = 1.0
        self.policy.actor_hidden_dims = [512, 256, 256]
        self.policy.critic_hidden_dims = [512, 256, 256]

        self.num_steps_per_env = 50
        self.max_iterations = 5000
        self.save_interval = 20
        self.clip_actions = 100.0
        self.empirical_normalization = False

        self.algorithm.learning_rate = 1.0e-3
        self.algorithm.entropy_coef = 0.01
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 4
        self.algorithm.gamma = 0.99
        self.algorithm.lam = 0.95
        self.algorithm.desired_kl = 0.01

        if RslRlSymmetryCfg is not None:
            self.algorithm.symmetry_cfg = RslRlSymmetryCfg(**maybe_get_symmetry_cfg_kwargs())
