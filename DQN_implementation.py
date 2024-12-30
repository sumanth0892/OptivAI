"""
Implementation of the paper HVAC_Control_Rl.pdf in Python, Torch and Stable_baselines
"""
import torch as th
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
import numpy as np


class CustomFeatureExtractor(nn.Module):
    """
    Custom feature extractor that processes different observation types
    """

    def __init__(self, observation_space):
        super().__init__()

        # Calculate total flattened size
        total_size = 0
        for key, space in observation_space.spaces.items():
            total_size += np.prod(space.shape)

        # Network as per paper architecture
        self.shared_net = nn.Sequential(
            nn.Linear(total_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

    def forward(self, observations) -> th.Tensor:
        # Flatten and concatenate all observations
        processed_obs = []

        for key, value in observations.items():
            processed_obs.append(value.flatten())

        combined = th.cat(processed_obs, dim=1)
        return self.shared_net(combined)


class ActionProcessor(gym.Wrapper):
    """
    Implements the action processor from the paper
    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        # Check if it's peak hours (9am-9pm weekdays)
        hour = self.env.current_time.hour
        weekday = self.env.current_time.weekday()

        if weekday < 5 and 9 <= hour <= 21:
            # During peak hours, use maximum comfortable temperature
            action = np.array([
                self.env.comfort_range['temp_low'],
                self.env.comfort_range['temp_high']
            ])

        return self.env.step(action)


class DemandChargeCallback(BaseCallback):
    """
    Implements reward shaping for demand charges
    """

    def __init__(self, power_threshold=0.6, penalty=100.0):
        super().__init__()
        self.power_threshold = power_threshold
        self.penalty = penalty

    def _on_step(self):
        # Get current power demand
        current_power = self.training_env.get_attr('current_power')[0][0]
        normalized_power = current_power / self.training_env.get_attr('max_cooling_power')[0]

        # Apply penalty if power exceeds threshold
        if normalized_power > self.power_threshold:
            self.locals['rewards'] -= self.penalty * (normalized_power - self.power_threshold)

        return True


class ModelSelectionCallback(EvalCallback):
    pass