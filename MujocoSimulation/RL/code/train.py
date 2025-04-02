import sys
import os
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,  # Save every 10,000 steps
    save_path="./ppo_checkpoints/",  # Folder to save the model
    name_prefix="ppo_model",  # Model file prefix
)


sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

print(sys.path)
from snake_env.envs.single_snake_world import SingleModuleWorldEnv

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

env_name = "SingleModuleWorldEnv-v0"
# Create and vectorize the environment
# env = gym.make("Humanoid-v5")
env = gym.make(env_name)

# env = DummyVecEnv([lambda: env])  # Needed for SB3

# Instantiate the PPO agent
# model = PPO("MlpPolicy",env, ent_coef=0.1, verbose=2)
model = PPO("MlpPolicy",env,n_steps=2048, batch_size=64, verbose=2,tensorboard_log="ppo_tensorboard")
model.learn(total_timesteps=10_000_000,callback=checkpoint_callback)

# Save the trained model
model.save("ppo_snake")
# Load the trained model
# model = PPO.load("ppo_snake")
#
# env = gym.make(env_name, render_mode="human")
# obs, info = env.reset()
# # for _ in range(1000):
# #     action, _ = model.predict(obs)
# #     obs, reward, done, _ = env.step(action)
# #     env.render()
# for _ in range(1000):
#     action, _ = model.predict(obs)
#     obs, reward, done, _, _ = env.step(action)  # ✅ Fix: Use 5-return Gymnasium API
#     env.render()
