from gymnasium.envs.registration import register

register(
    id="SingleModuleWorldEnv-v0",
    entry_point="snake_env.envs.single_snake_world:SingleModuleWorldEnv",
)
