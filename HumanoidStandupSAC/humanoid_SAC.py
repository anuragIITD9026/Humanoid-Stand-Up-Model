import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import matplotlib.pyplot as plt

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1)
        )

    def size(self):
        return len(self.buffer)
        
# Running Mean/Std Normalizer
class Normalizer:
    def __init__(self, size, eps=1e-8, clip=10.0):
        self.size = size
        self.eps = eps
        self.clip = clip
        self.mean = np.zeros(size, dtype=np.float32)
        self.var = np.ones(size, dtype=np.float32)
        self.count = eps

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x, update_stats=True):
        #Normalize observations
        if update_stats:
            if x.ndim == 1:
                self.update(x.reshape(1, -1))
            else:
                self.update(x)
        
        normalized = (x - self.mean) / np.sqrt(self.var + self.eps)
        return np.clip(normalized, -self.clip, self.clip)

# Reward Shaping Wrapper 
class HumanoidStandupRewardShaping(gym.Wrapper):
  
    def __init__(self, env, shaping_weight=0.1):
        super().__init__(env)
        self.shaping_weight = shaping_weight
        self.prev_height = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_height = None
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract torso height 
        z_pos = obs[0]
        
        # Extract orientation 
        quat = obs[3:7] if len(obs) > 6 else [1, 0, 0, 0]
        quat = quat / (np.linalg.norm(quat) + 1e-8)
        upright_score = abs(quat[0])
        
        # Shaped rewards
        shaped_reward = 0.0
        
        # Height progress reward
        if self.prev_height is not None:
            height_progress = z_pos - self.prev_height
            shaped_reward += height_progress * 10.0
        self.prev_height = z_pos
        
        # Upright posture reward
        shaped_reward += upright_score * 0.5
        
        # Action smoothness penalty
        action_penalty = -0.01 * np.sum(np.square(action))
        shaped_reward += action_penalty
        
        # Stability penalty when standing
        if z_pos > 1.0:
            velocity = obs[24:47] if len(obs) > 47 else []
            if len(velocity) > 0:
                velocity_penalty = -0.001 * np.sum(np.square(velocity))
                shaped_reward += velocity_penalty
        
        total_reward = reward + self.shaping_weight * shaped_reward
        
        info['original_reward'] = reward
        info['shaped_reward'] = shaped_reward
        
        return obs, total_reward, terminated, truncated, info

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Better initialization
        nn.init.uniform_(self.mean.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std.bias, -3e-3, 3e-3)

    def forward(self, x):
        x = self.net(x)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

# SAC Agent 
class SACAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.005, 
                 alpha=0.2, auto_alpha=True, lr=3e-4, hidden_dim=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        
        # Automatic temperature tuning
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            mean, _ = self.actor(state)
            action = torch.tanh(mean)
        else:
            action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        if replay_buffer.size() < batch_size:
            return {}

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # Update Critic
        with torch.no_grad():
            next_action, log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.target_critic(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update Actor
        action_pi, log_pi = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, action_pi)
        actor_loss = (self.alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Update temperature (alpha)
        alpha_loss = 0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft update target critic
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss.item() if self.auto_alpha else 0
        }

# Training Loop
def train(use_reward_shaping=False, shaping_weight=0.1):
    # Environment setup
    env = gym.make("HumanoidStandup-v5")
    
    if use_reward_shaping:
        print(f" Using reward shaping with weight {shaping_weight}")
        env = HumanoidStandupRewardShaping(env, shaping_weight=shaping_weight)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Agent and buffer
    agent = SACAgent(state_dim, action_dim, auto_alpha=True, lr=3e-4, hidden_dim=256)
    replay_buffer = ReplayBuffer(max_size=1000000)
    normalizer = Normalizer(state_dim)

    # Training parameters
    max_steps = 3_000_000
    warmup_steps = 10000
    eval_freq = 10000
    batch_size = 256
    updates_per_step = 1
    
    # Logging
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    episode_rewards = []
    eval_rewards = []
    total_steps = 0
    episode = 0

    state, _ = env.reset()
    state = normalizer.normalize(state, update_stats=True)
    episode_reward = 0
    episode_steps = 0

    print("\n Starting training...\n")

    while total_steps < max_steps:
        # Select action
        if total_steps < warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        # Execute action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = normalizer.normalize(next_state, update_stats=True)

        # Store transition
        replay_buffer.store(state, action, reward, next_state, float(done))

        state = next_state
        episode_reward += reward
        episode_steps += 1
        total_steps += 1

        # Update agent
        if total_steps >= warmup_steps:
            for _ in range(updates_per_step):
                metrics = agent.update(replay_buffer, batch_size)

        # Handle episode end
        if done:
            episode += 1
            episode_rewards.append(episode_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode} | Steps: {total_steps} | "
                      f"Reward: {episode_reward:.1f} | Avg(10): {avg_reward:.1f} | "
                      f"Alpha: {agent.alpha:.3f}")
            
            state, _ = env.reset()
            state = normalizer.normalize(state, update_stats=True)
            episode_reward = 0
            episode_steps = 0

        # Evaluation
        if total_steps % eval_freq == 0 and total_steps >= warmup_steps:
            eval_reward = evaluate(env, agent, normalizer, num_episodes=5)
            eval_rewards.append((total_steps, eval_reward))
            print(f"\nEvaluation at step {total_steps}: {eval_reward:.1f}\n")
            
            # Save checkpoint
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'target_critic': agent.target_critic.state_dict(),
                'actor_optimizer': agent.actor_optimizer.state_dict(),
                'critic_optimizer': agent.critic_optimizer.state_dict(),
                'total_steps': total_steps,
                'normalizer_mean': normalizer.mean,
                'normalizer_var': normalizer.var,
            }, f"{checkpoint_dir}/checkpoint_{total_steps}.pth")

    env.close()

    # Plot results
    plot_results(episode_rewards, eval_rewards)
    print("\nTraining complete!")
    
    return agent, normalizer

def evaluate(env, agent, normalizer, num_episodes=10):
    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        state = normalizer.normalize(state, update_stats=False)
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = normalizer.normalize(next_state, update_stats=False)
            episode_reward += reward
            
        rewards.append(episode_reward)
    
    return np.mean(rewards)

def plot_results(episode_rewards, eval_rewards):
    #Plot and save training results
    os.makedirs("results", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Episode rewards
    ax1.plot(episode_rewards, alpha=0.3, color='royalblue', label='Episode Reward')
    if len(episode_rewards) >= 100:
        smoothed = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        ax1.plot(range(99, len(episode_rewards)), smoothed, color='darkblue', 
                linewidth=2, label='100-ep average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Evaluation rewards
    if eval_rewards:
        steps, rewards = zip(*eval_rewards)
        ax2.plot(steps, rewards, marker='o', color='green', linewidth=2, 
                markersize=6, label='Evaluation Reward')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Evaluation Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/training_results.png", dpi=300)
    plt.close()
    print("Results saved to results/training_results.png")
def load_checkpoint(agent, normalizer, checkpoint_path):
    #Load a saved checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])
    agent.target_critic.load_state_dict(checkpoint['target_critic'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
    normalizer.mean = checkpoint['normalizer_mean']
    normalizer.var = checkpoint['normalizer_var']
    total_steps = checkpoint['total_steps']
    print(f" Checkpoint loaded from step {total_steps}")
    return total_steps


def test_agent(checkpoint_path=None, num_episodes=3, render=True):
    
    env = gym.make("HumanoidStandup-v5", render_mode="human" if render else None)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SACAgent(state_dim, action_dim)
    normalizer = Normalizer(state_dim)
    
    if checkpoint_path:
        load_checkpoint(agent, normalizer, checkpoint_path)
    
    rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        state = normalizer.normalize(state, update_stats=False)
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = normalizer.normalize(next_state, update_stats=False)
            episode_reward += reward
            steps += 1
            
        rewards.append(episode_reward)
        print(f"Episode {ep+1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    env.close()
    print(f"\nAverage Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

if __name__ == "__main__":
    # Training mode
    #agent, normalizer = train(use_reward_shaping=False, shaping_weight=0.1)
        # To test a trained agent (uncomment and provide checkpoint path)
    test_agent("checkpoints/checkpoint_3000000.pth")

