import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

print(sys.path)
from snake_env.envs.single_snake_world import SingleModuleWorldEnv

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

env_name = "SingleModuleWorldEnv-v0"
model = PPO.load("ppo_checkpoints/ppo_model_9600000_steps")

env = gym.make(env_name, render_mode="human")
obs, info = env.reset()
# for _ in range(1000):
#     action, _ = model.predict(obs)
#     obs, reward, done, _ = env.step(action)
#     env.render()
# Define the size of the dataset
num_entries = 10000
num_features = 5
data = np.zeros((num_entries, num_features))
data2 = np.zeros((num_entries, num_features))

for i in range(10000):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)  # ✅ Fix: Use 5-return Gymnasium API
    # print("reward:", reward)
    print("action:", action)
    data[i] = action
    data2[i] = obs[6:]
    env.render()
#######################
timesteps = np.arange(num_entries)
# Create subplots for each feature
fig, axes = plt.subplots(num_features, 1, figsize=(10, 10), sharex=True)
fig.suptitle('Action Data Over Time')

for i in range(num_features):
    axes[i].plot(timesteps, data[:, i], label=f'Feature {i+1}')
    axes[i].plot(timesteps, data2[:, i], label=f'Joint Sensor {i+6}', linestyle='dashed', color='r')

    axes[i].set_ylabel(f'Feature {i+1}')
    axes[i].legend()
    axes[i].grid(True)

axes[-1].set_xlabel('Timestep')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#######################

# plt.figure(figsize=(10, 6))
# timesteps = np.arange(num_entries)
# for i in range(num_features):
#     plt.plot(timesteps, data[:, i], label=f'Feature {i+1}')
#
# plt.xlabel('Timestep')
# plt.ylabel('Values')
# plt.title('Action Data Over Time')
# plt.legend()
# plt.grid(True)
# plt.show()
#######################
