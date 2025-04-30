import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make("CustomEnv-v1", render_mode="human")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()

class CustomEnv(MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file: str = "ant.xml",
        ...
        **kwargs,
    ):
        MujocoEnv.__init__(
            self,
            xml_file,
            ...
            **kwargs,
        )0

    def step(self, action):
        ...
        self.do_simulation(action, self.frame_skip)
        ...
        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity: float, action):
        ...
        return reward, reward_info

    def _get_obs(self):
        return obs

    def reset(self):
        ...
        return obs

    def close(self):
        ...
