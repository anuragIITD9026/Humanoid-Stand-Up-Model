
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

def train():
    #device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
#defining result and log file path
    log_dir = "./logs"
    model_path = os.path.join(log_dir, "ppo_humanoid_standup.zip")
    os.makedirs(log_dir, exist_ok=True)

    # setting parallel nvironment for fast training
    n_envs = 4
    env = make_vec_env(
        "HumanoidStandup-v4",
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    eval_env = make_vec_env("HumanoidStandup-v4", n_envs=1)
    vecnorm_path = os.path.join(log_dir, "vecnormalize_latest.pkl")
    if os.path.exists(vecnorm_path):
        eval_env = VecNormalize.load(vecnorm_path, eval_env)
    else:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    eval_env.training = False
    eval_env.norm_reward = False
    # checkpoint callback
    class CheckpointCallback(BaseCallback):
        def __init__(self, save_freq, save_path, verbose=1):
            super().__init__(verbose)
            self.save_freq = save_freq
            self.save_path = save_path

        def _on_step(self) -> bool:
            if self.n_calls % self.save_freq == 0:
                save_file = os.path.join(self.save_path, "ppo_humanoid_standup_latest.zip")
                self.model.save(save_file)
                self.training_env.save(os.path.join(self.save_path, "vecnormalize_latest.pkl"))
                if self.verbose > 0:
                    print(f"[INFO] Checkpoint saved at step {self.num_timesteps} -> {save_file}")
            return True

    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=log_dir)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=50_000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    callback = CallbackList([eval_callback, checkpoint_callback])
    # Policy network
    policy_kwargs = dict(
        net_arch=[512, 256, 128],
        activation_fn=torch.nn.ReLU
    )

    # Load existing model or create new
    if os.path.exists(model_path):
        print("[INFO] Loading existing model...")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True
        env.norm_reward = True
        model = PPO.load(model_path, env=env, device=device)
    else:
        print("[INFO] Creating new model...")
        model = PPO(
            "MlpPolicy",
            env,
            device=device,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=lambda f: 3e-4 * f,  
            n_steps=2048,
            batch_size=4096,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
        )

    # Training
    print("[INFO] Starting training...")
    model.learn(
        total_timesteps=25_000_000,  
        callback=callback,
        tb_log_name="ppo_humanoid_standup",
        progress_bar=True
    )
    # Save final model
    model.save(model_path)
    env.save(os.path.join(log_dir, "vecnormalize_final.pkl"))
    print("[INFO] Training completed.")

if __name__ == "__main__":
    train()
