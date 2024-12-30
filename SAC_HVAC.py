"""
Python and Torch implementation of the paper Experimental_evaluation_HVAC_control_RL.pdf
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import collections
import random


# Neural Networks for Actor and Critic
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # Prevent extreme values
        return mean, log_std


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        # Q1
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)

        # Q2
        q2 = F.relu(self.fc3(x))
        q2 = F.relu(self.fc4(q2))
        q2 = self.q2(q2)

        return q1, q2


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)


class SAC:
    def __init__(self, state_dim, action_dim, action_space_low, action_space_high):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        # Copy critic parameters to target
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.replay_buffer = ReplayBuffer(1000000)
        self.batch_size = 256

        self.gamma = 0.99  # Discount factor
        self.tau = 0.005  # Soft update parameter
        self.alpha = 0.2  # Temperature parameter

        self.action_scale = torch.FloatTensor(
            (action_space_high - action_space_low) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_space_high + action_space_low) / 2.)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.actor(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        x_t = normal.rsample()  # Re-parameterization trick
        action = torch.tanh(x_t)
        action = action * self.action_scale + self.action_bias
        return action.detach().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_buffer.sample(self.batch_size)

        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1)

        # Update critic
        with torch.no_grad():
            next_mean, next_log_std = self.actor(next_state_batch)
            next_std = next_log_std.exp()
            next_normal = Normal(next_mean, next_std)
            next_action = torch.tanh(next_normal.rsample())
            next_action = next_action * self.action_scale + self.action_bias

            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        mean, log_std = self.actor(state_batch)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        action = action * self.action_scale + self.action_bias

        q1, q2 = self.critic(state_batch, action)
        q = torch.min(q1, q2)

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        actor_loss = (self.alpha * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update critic target
        for param, target_param in zip(self.critic.parameters(),
                                       self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


# Example usage
if __name__ == "__main__":
    # Example state dimension (temperature, humidity, time, etc.)
    state_dim = 5
    # Example action dimension (heating/cooling setpoints)
    action_dim = 2

    # Action space bounds for temperature setpoints
    action_space_low = np.array([20.0, 20.0])  # Min temp setpoints
    action_space_high = np.array([26.0, 26.0])  # Max temp setpoints

    # Initialize SAC agent
    agent = SAC(state_dim, action_dim, action_space_low, action_space_high)

    # Example state (temp, humidity, time, outdoor temp, energy consumption)
    state = np.array([22.0, 0.5, 0.25, 25.0, 0.8])

    # Get action from agent
    action = agent.select_action(state)
    print(f"For state: {state}")
    print(f"Agent recommended setpoints: {action}")

    # Example of updating agent with some fake experience
    reward = -1.0  # Example reward combining energy and comfort
    next_state = np.array([22.5, 0.48, 0.26, 25.2, 0.75])
    done = False

    # Store experience in replay buffer
    agent.replay_buffer.push(state, action, reward, next_state, done)

    # Update networks
    agent.update()