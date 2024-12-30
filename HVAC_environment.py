"""
Physical environment simulation for the setup in the paper HVAC_Control_RL.pdf
This sets up a playground of sorts and we need to fill it with various rules and data for the agent to train and
understand the scenario.
"""
import gym
from gym import spaces
import numpy as np
from datetime import datetime, timedelta
import math


class DetailedHVACEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Define observation space according to paper
        self.observation_space = spaces.Dict({
            # Core measurements
            'zone_temp': spaces.Box(low=15.0, high=30.0, shape=(1,), dtype=np.float32),
            'outdoor_temp': spaces.Box(low=-20.0, high=50.0, shape=(1,), dtype=np.float32),
            'zone_humidity': spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            'outdoor_humidity': spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),

            # Weather forecasts (12-hour horizon, 15-min intervals = 48 points)
            'temp_forecast': spaces.Box(low=-20.0, high=50.0, shape=(48,), dtype=np.float32),
            'humidity_forecast': spaces.Box(low=0.0, high=100.0, shape=(48,), dtype=np.float32),

            # Power and energy
            'current_power': spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            'hvac_power': spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32),
            'base_load': spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32),
            'peak_demand': spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),

            # Time features
            'hour': spaces.Box(low=0, high=23, shape=(1,), dtype=np.int32),
            'day': spaces.Box(low=0, high=6, shape=(1,), dtype=np.int32),
            'month': spaces.Box(low=0, high=11, shape=(1,), dtype=np.int32)
        })

        # Action space: heating and cooling setpoints
        self.action_space = spaces.Box(
            low=np.array([15.0, 15.0], dtype=np.float32),  # [heating_setpoint, cooling_setpoint]
            high=np.array([30.0, 30.0], dtype=np.float32)
        )

        # Building physics parameters
        self.thermal_mass = 2000.0  # kJ/K
        self.heat_transfer_coefficient = 100.0  # W/K
        self.solar_gain_coefficient = 0.3
        self.infiltration_rate = 0.5  # Air changes per hour
        self.room_volume = 1000.0  # m³

        # HVAC system parameters
        self.cop_cooling = 3.0  # Coefficient of Performance for cooling
        self.cop_heating = 2.5  # Coefficient of Performance for heating
        self.max_cooling_power = 20000.0  # W
        self.max_heating_power = 15000.0  # W

        # Comfort parameters
        self.comfort_range = {
            'temp_low': 22.0,
            'temp_high': 24.0,
            'humidity_low': 30.0,
            'humidity_high': 60.0
        }

        # Cost parameters (from paper)
        self.electricity_prices = {
            'peak': 0.20101,  # $/kWh during peak hours (9am-9pm weekdays)
            'off_peak': 0.09137  # $/kWh during off-peak hours
        }
        self.demand_charge = 15.65  # $/kW monthly peak demand charge

        # Initialize simulation time
        self.current_time = datetime(2024, 1, 1)
        self.timestep = timedelta(minutes=15)  # 15-minute timesteps

        # Initialize state
        self.reset()

    def _calculate_solar_gain(self):
        """Calculate solar heat gain based on time of day."""
        hour = self.current_time.hour
        # Simple solar gain model based on time of day
        if 6 <= hour <= 18:  # Daylight hours
            return self.solar_gain_coefficient * math.sin(math.pi * (hour - 6) / 12) * 1000
        return 0.0

    def _calculate_internal_gains(self):
        """Calculate internal heat gains from occupancy and equipment."""
        hour = self.current_time.hour
        weekday = self.current_time.weekday()

        # Base load
        base_load = 500.0  # W

        # Occupancy and equipment schedule
        if weekday < 5 and 8 <= hour <= 17:  # Weekday working hours
            return base_load + 1500.0  # Additional load during occupied hours
        return base_load

    def _calculate_hvac_power(self, setpoints, current_temp):
        """Calculate HVAC power consumption based on setpoints and current temperature."""
        heating_setpoint, cooling_setpoint = setpoints

        if current_temp < heating_setpoint:
            # Heating mode
            delta_t = heating_setpoint - current_temp
            power_needed = min(self.max_heating_power, delta_t * self.heat_transfer_coefficient)
            return power_needed / self.cop_heating

        elif current_temp > cooling_setpoint:
            # Cooling mode
            delta_t = current_temp - cooling_setpoint
            power_needed = min(self.max_cooling_power, delta_t * self.heat_transfer_coefficient)
            return power_needed / self.cop_cooling

        return 0.0  # No HVAC needed

    def _update_temperature(self, hvac_power, outdoor_temp):
        """Update zone temperature based on building physics."""
        current_temp = self.state['zone_temp'][0]

        # Heat flows
        solar_gain = self._calculate_solar_gain()
        internal_gains = self._calculate_internal_gains()
        transmission_loss = self.heat_transfer_coefficient * (current_temp - outdoor_temp)
        infiltration_loss = self.infiltration_rate * self.room_volume * 1.2 * 1000 * (
                    current_temp - outdoor_temp) / 3600

        # Net heat flow
        net_heat_flow = solar_gain + internal_gains - transmission_loss - infiltration_loss + hvac_power

        # Temperature change
        delta_t = (net_heat_flow * self.timestep.total_seconds()) / self.thermal_mass

        return current_temp + delta_t

    def step(self, action):
        # Validate set-points
        heating_setpoint, cooling_setpoint = action
        if heating_setpoint > cooling_setpoint:
            heating_setpoint = cooling_setpoint - 1.0  # Ensure valid set point range

        # Get current conditions
        current_temp = self.state['zone_temp'][0]
        outdoor_temp = self.state['outdoor_temp'][0]

        # Calculate HVAC response and power
        hvac_power = self._calculate_hvac_power([heating_setpoint, cooling_setpoint], current_temp)

        # Update temperature
        new_temp = self._update_temperature(hvac_power, outdoor_temp)

        # Calculate costs
        hour = self.current_time.hour
        weekday = self.current_time.weekday()
        is_peak = weekday < 5 and 9 <= hour <= 21

        energy_price = self.electricity_prices['peak'] if is_peak else self.electricity_prices['off_peak']
        energy_cost = (hvac_power / 1000.0) * energy_price * (self.timestep.total_seconds() / 3600.0)

        # Update peak demand if necessary
        current_demand = (hvac_power + self.state['base_load'][0]) / 1000.0  # kW
        peak_demand = max(self.state['peak_demand'][0], current_demand)

        # Calculate comfort violation
        temp_violation = max(0, self.comfort_range['temp_low'] - new_temp) + \
                         max(0, new_temp - self.comfort_range['temp_high'])

        # Calculate reward
        lambda_p = 0.00001  # Power scaling factor
        lambda_c = 0.1  # Comfort scaling factor
        omega = 0.5  # Weight between power and comfort

        reward = -(
                omega * lambda_p * energy_cost +
                (1 - omega) * lambda_c * temp_violation +
                self.demand_charge * (peak_demand - self.state['peak_demand'][0]) / 30.0
        )

        # Update state
        self.state['zone_temp'] = np.array([new_temp])
        self.state['hvac_power'] = np.array([hvac_power])
        self.state['current_power'] = np.array([current_demand * 1000.0])
        self.state['peak_demand'] = np.array([peak_demand])

        # Update time
        self.current_time += self.timestep
        self.state['hour'] = np.array([self.current_time.hour])
        self.state['day'] = np.array([self.current_time.weekday()])
        self.state['month'] = np.array([self.current_time.month - 1])

        # Update weather (simplified)
        self._update_weather()

        done = False  # Episodes could end at month boundaries
        info = {
            'temp_violation': temp_violation,
            'energy_cost': energy_cost,
            'peak_demand': peak_demand
        }

        return self.state, reward, done, info

    def _update_weather(self):
        """Update weather conditions (simplified)."""
        hour = self.current_time.hour

        # Simple sinusoidal outdoor temperature variation
        base_temp = 20.0
        amplitude = 5.0
        outdoor_temp = base_temp + amplitude * math.sin(2 * math.pi * (hour - 4) / 24)
        outdoor_temp += np.random.normal(0, 0.5)  # Add some noise

        self.state['outdoor_temp'] = np.array([outdoor_temp])

        # Update forecasts (simplified)
        forecasts = []
        for i in range(48):  # 12 hours * 4 (15-min intervals)
            forecast_hour = (hour + i / 4) % 24
            forecast_temp = base_temp + amplitude * math.sin(2 * math.pi * (forecast_hour - 4) / 24)
            forecasts.append(forecast_temp)

        self.state['temp_forecast'] = np.array(forecasts)

    def reset(self):
        """Reset environment state."""
        self.current_time = datetime(2024, 1, 1)

        self.state = {
            'zone_temp': np.array([22.0]),
            'outdoor_temp': np.array([20.0]),
            'zone_humidity': np.array([45.0]),
            'outdoor_humidity': np.array([50.0]),
            'temp_forecast': np.zeros(48),
            'humidity_forecast': np.zeros(48),
            'current_power': np.array([0.0]),
            'hvac_power': np.array([0.0]),
            'base_load': np.array([5000.0]),  # Base electrical load
            'peak_demand': np.array([0.0]),
            'hour': np.array([0]),
            'day': np.array([0]),
            'month': np.array([0])
        }

        self._update_weather()
        return self.state