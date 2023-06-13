from gym.envs.registration import register

register(
     id="StormEnv-v0",
     entry_point="stormbot.gym_environments.storm_env_v0:StormEnv_v0",
)

register(
     id="StormEnv-v1",
     entry_point="stormbot.gym_environments.storm_env_v1:StormEnv_v1",
)

register(
     id="StormEnv-v2",
     entry_point="stormbot.gym_environments.storm_env_v2:StormEnv_v2",
)

register(
     id="SlimboundEnv",
     entry_point="stormbot.gym_environments.slimbound_env:SlimboundEnv",
)

# register(
#      id="SlimboundEnv-2",
#      entry_point="stormbot.gym_environments.slimbound_env_2:SlimboundEnv_2",
# )