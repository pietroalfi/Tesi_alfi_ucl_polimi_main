from gym.envs.registration import register

register(
    id='markets-execution-v1',  # Unique identifier for your environment
    entry_point='abides_gym.envs.markets_execution_environment_v1:SubGymMarketsExecutionEnv_v1',  # Module and class where your environment is defined
)