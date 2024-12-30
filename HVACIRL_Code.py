"""
Code from the paper HVACIRL_Preprint.pdf
The study from that paper is implemented here in Python and Sinergym
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import sinergym
import gym
from collections import deque, namedtuple
import random
import wandb  # For experiment tracking
import yaml
import logging
from datetime import datetime
import os
import pandas as pd
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Replay buffer for storing and sampling experiences"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class CustomReward:
    """Custom reward shaping for HVAC control"""

    def __init__(self, comfort_weight: float = 0.5, power_weight: float = 0.5):
        self.comfort_weight = comfort_weight
        self.power_weight = power_weight

    def calculate_reward(self,
                         power_consumption: float,
                         current_temp: float,
                         target_temp: float,
                         comfort_range: Tuple[float, float]) -> float:
        """Calculate reward based on power consumption and comfort"""
        # Power consumption penalty
        power_penalty = -self.power_weight * power_consumption

        # Comfort violation penalty
        temp_violation = 0
        if current_temp < comfort_range[0]:
            temp_violation = comfort_range[0] - current_temp
        elif current_temp > comfort_range[1]:
            temp_violation = current_temp - comfort_range[1]

        comfort_penalty = -self.comfort_weight * (temp_violation ** 2)

        # Combine rewards
        total_reward = power_penalty + comfort_penalty

        return total_reward


class ActorCritic(nn.Module):
    """Enhanced ActorCritic network with additional features"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Actor network
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # State-dependent log standard deviation
        self.actor_log_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(state)

        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std(features)
        action_std = torch.exp(action_log_std)

        value = self.critic(features)

        return action_mean, action_std, value


class HVACController:
    """Main HVAC controller implementing both BC and PPO"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize environment
        self.env = sinergym.make(config['env_name'])
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # Initialize networks
        self.actor_critic = ActorCritic(
            self.state_dim,
            self.action_dim,
            config['hidden_dim']
        ).to(self.device)

        # Initialize optimizers
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=config['learning_rate']
        )

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(config['buffer_capacity'])

        # Initialize reward shaper
        self.reward_shaper = CustomReward(
            config['comfort_weight'],
            config['power_weight']
        )

        # Setup logging directory
        self.setup_logging()

    def setup_logging(self):
        """Setup logging directory and wandb"""
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = os.path.join('logs', current_time)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize wandb
        if self.config['use_wandb']:
            wandb.init(
                project="hvac_control",
                config=self.config,
                name=f"run_{current_time}"
            )

    def behavioral_cloning(self, expert_data: pd.DataFrame):
        """Pre-training using behavioral cloning"""
        logger.info("Starting behavioral cloning pre-training...")

        for epoch in range(self.config['bc_epochs']):
            total_loss = 0

            for batch in self.get_expert_batches(expert_data):
                states = torch.FloatTensor(batch['states']).to(self.device)
                actions = torch.FloatTensor(batch['actions']).to(self.device)

                # Forward pass
                action_mean, action_std, _ = self.actor_critic(states)
                dist = Normal(action_mean, action_std)

                # Calculate loss
                log_prob = dist.log_prob(actions)
                loss = -log_prob.mean()

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # Log metrics
            avg_loss = total_loss / len(expert_data)
            logger.info(f"BC Epoch {epoch}: Average Loss = {avg_loss:.4f}")

            if self.config['use_wandb']:
                wandb.log({'bc_loss': avg_loss})

    def ppo_update(self, memory: List[Experience]):
        """Update policy using PPO"""
        # Convert experiences to tensors
        states = torch.FloatTensor([e.state for e in memory]).to(self.device)
        actions = torch.FloatTensor([e.action for e in memory]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in memory]).to(self.device)

        # Get old action probabilities
        with torch.no_grad():
            old_action_mean, old_action_std, old_values = self.actor_critic(states)
            old_dist = Normal(old_action_mean, old_action_std)
            old_log_probs = old_dist.log_prob(actions)

        # PPO update for K epochs
        for _ in range(self.config['ppo_epochs']):
            # Get current action probabilities
            action_mean, action_std, values = self.actor_critic(states)
            dist = Normal(action_mean, action_std)
            log_probs = dist.log_prob(actions)

            # Calculate ratio and clipped surrogate objective
            ratios = torch.exp(log_probs - old_log_probs.detach())
            advantages = rewards - old_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(
                ratios,
                1 - self.config['clip_epsilon'],
                1 + self.config['clip_epsilon']
            ) * advantages

            # Calculate losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * ((values - rewards) ** 2).mean()
            entropy_loss = -dist.entropy().mean()

            total_loss = (
                    actor_loss +
                    self.config['value_coef'] * critic_loss +
                    self.config['entropy_coef'] * entropy_loss
            )

            # Update networks
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Log metrics
            if self.config['use_wandb']:
                wandb.log({
                    'actor_loss': actor_loss.item(),
                    'critic_loss': critic_loss.item(),
                    'entropy_loss': entropy_loss.item()
                })

    def train(self):
        """Main training loop"""
        logger.info("Starting training...")

        episode_rewards = []

        for episode in range(self.config['num_episodes']):
            state = self.env.reset()
            episode_reward = 0

            for step in range(self.config['max_steps']):
                # Select action
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    action_mean, action_std, _ = self.actor_critic(state_tensor)
                    dist = Normal(action_mean, action_std)
                    action = dist.sample()

                # Take action in environment
                next_state, reward, done, info = self.env.step(action.cpu().numpy())

                # Shape reward
                shaped_reward = self.reward_shaper.calculate_reward(
                    info['power_consumption'],
                    info['current_temp'],
                    info['target_temp'],
                    self.config['comfort_range']
                )

                # Store experience
                self.replay_buffer.push(state, action, shaped_reward, next_state, done)

                episode_reward += shaped_reward
                state = next_state

                # Update policy if enough experiences are collected
                if len(self.replay_buffer) >= self.config['batch_size']:
                    experiences = self.replay_buffer.sample(self.config['batch_size'])
                    self.ppo_update(experiences)

                if done:
                    break

            # Log episode metrics
            episode_rewards.append(episode_reward)
            mean_reward = np.mean(episode_rewards[-100:])

            logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}, "
                        f"Mean Reward = {mean_reward:.2f}")

            if self.config['use_wandb']:
                wandb.log({
                    'episode_reward': episode_reward,
                    'mean_reward': mean_reward
                })

            # Save model checkpoint
            if episode % self.config['save_freq'] == 0:
                self.save_checkpoint(episode)

    def save_checkpoint(self, episode: int):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.log_dir,
            f'checkpoint_episode_{episode}.pt'
        )
        torch.save({
            'episode': episode,
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def get_expert_batches(self, expert_data: pd.DataFrame):
        """Generate batches from expert data"""
        # Implementation depends on expert data format
        pass


def main():
    # Load configuration
    with open('config_HVACIRL.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize controller
    controller = HVACController(config)

    # Load expert data for behavioral cloning
    expert_data = pd.read_csv(config['expert_data_path'])

    # Pre-train with behavioral cloning
    controller.behavioral_cloning(expert_data)

    # Fine-tune with PPO
    controller.train()


if __name__ == '__main__':
    main()