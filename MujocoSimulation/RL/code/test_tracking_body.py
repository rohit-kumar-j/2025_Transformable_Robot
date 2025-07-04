import sys
import os
import numpy as np
import mujoco
import matplotlib.pyplot as plt

import gymnasium as gym

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

print(sys.path)
from snake_env.envs.single_snake_world import SingleModuleWorldEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

env_name = "SingleModuleWorldEnv-v0"
model = PPO.load("ppo_checkpoints/ppo_model_8400000_steps")

env = gym.make(env_name, render_mode="human")
env = env.unwrapped
env.render()
viewer = env.mujoco_renderer.viewer  # Access the viewer to control the camera
cam_id  = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, 'tracking_cam')

# Set the camera to track this body
viewer.cam.trackbodyid = cam_id

# To create graph
enable_logging = False
num_entries = 10000
num_features = 5
data = np.zeros((num_entries, num_features))
data2 = np.zeros((num_entries, num_features))

obs, info = env.reset()
for i in range(10000):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)  # Fix: Use 5-return Gymnasium API

    # print("reward:", reward)
    print("action:", action)
    if(enable_logging):
        data[i] = action
        data2[i] = obs[6:]

    # Get center of mass (CoM) of the entire system
    com_position = np.array(env.unwrapped.data.subtree_com[0])  # Root body CoM

    # Update camera look-at position to track CoM
    viewer.cam.lookat[:] = com_position  # Update the camera's lookat point
    viewer.cam.distance = 10  # Adjust distance for a better view (can be dynamic based on needs)

    env.render()
#######################
if(enable_logging):
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

env.close()
