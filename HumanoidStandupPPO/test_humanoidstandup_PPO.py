
import os
import torch
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def test():
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    # log folder Path
    log_dir = "./logs"
    model_path = os.path.join(log_dir, "ppo_humanoid_standup.zip")
    vecnorm_path = os.path.join(log_dir, "vecnormalize_final.pkl")
    # Load environment
    env = gym.make("HumanoidStandup-v4", render_mode="human")  # render_mode="human" to visualize
    env = DummyVecEnv([lambda: env])

    # Load normalization statistics
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        print("[INFO] Loaded VecNormalize stats.")
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        print("[WARN] VecNormalize stats not found. Using default normalization.")

    env.training = False
    env.norm_reward = False

    print("[INFO] Loading trained model...")
    model = PPO.load(model_path, env=env, device=device)

    num_episodes = 2
    total_rewards = []
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            time.sleep(0.01)  

        total_rewards.append(ep_reward)
        
        print(f"[INFO] Episode {ep+1} Reward: {ep_reward}")

    avg_reward = sum(total_rewards) / num_episodes
    print(f"\n [RESULT] Average Reward over {num_episodes} episodes: {avg_reward}")

    env.close()

if __name__ == "__main__":
    test()
