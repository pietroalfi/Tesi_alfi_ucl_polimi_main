import ray
from ray import tune
# Rimuovi/Commenta le import relative a wandb
# from ray.tune.integration.wandb import WandbLoggerCallback
# import wandb

from abides_gym.envs.markets_data_environment import SubGymGetInfoMkt
from abides_gym.envs.markets_analysis_custom_metrics import MyBaseCallbacksMkt
from ray.tune.registry import register_env

# Registra l'ambiente personalizzato
register_env(
    "mkt-data-v0",
    lambda config: SubGymGetInfoMkt(**config),
)


ray.shutdown()
ray.init(num_cpus=2)

tune.run(
    "DQN",
    name="dqn_without_wandb",
    resume=False,
    stop={"training_iteration": 17},
    checkpoint_at_end=True,
    checkpoint_freq=10,
    config={
        # Environment
        "env": "mkt-data-v0",
        "env_config": {
            "background_config": "intel_basic",
            "timestep_duration": "0.2s",
            "state_timesteps_back": 0,
            "dirichlet_steps_ahead": 8,
            "dirichlet_k": 0.00005,
            "dirichlet_a_h": 5.0,
            "dirichlet_a_l": 1.0,
            "starting_cash": 2_000_000,
            "max_holdings": 10,
            "order_fixed_size": 1,
            "state_history_length": 2,
            "market_data_buffer_length": 5,
            "w_direc": 1,
            "reduction_factor": 0.99996,
            "kappa": 5,
            "debug_mode": True,
            "first_interval": "00:00:30",
            "background_config_extra_kvargs": {
                "ticker": "INTC",
                "period_starts": ['20241001 120000', '20241002 120000', '20241003 120000', '20241004 120000',
                                  '20241007 120000', '20241008 120000', '20241009 120000', '20241010 120000',
                                  '20241011 120000', '20241014 120000', '20241015 120000', '20241016 120000',
                                  '20241017 120000', '20241018 120000', '20241021 120000', '20241022 120000',
                                  '20241023 120000', '20241024 120000', '20241025 120000', '20241028 120000',
                                  '20241029 120000', '20241030 120000', '20241031 120000', '20241101 120000',
                                  '20241104 120000', '20241105 120000', '20241106 120000', '20241107 120000',
                                  '20241108 120000', '20241111 120000', '20241112 120000', '20241113 120000', 
                                  '20241114 120000', '20241115 120000', '20241118 120000'],
                "fundamental_file_path": "/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Merged_data/INTC_merged_mid_price_train.pkl",
                "exchange_log_orders": False,
                "log_orders": False,
                "book_logging": False,
                "book_log_depth": 10,
                "starting_cash": 200_000,
                "seed": 1,
                "num_noise_agents": 5000,
                "num_momentum_agents": 40,
                "m_wake_up_freq": "0.5s",
                "num_value_agents": 200,
                "val_wake_up_freq": "60s",
                "val_lambda_a": 1e-9,#7e-11,
                "num_mm": 2,
                "mm_wake_up_freq": "0.1s",
                "mm_window_size": "adaptive",
                "mm_pov": 0.025,
                "mm_num_ticks": 5,
                "mm_min_order_size": 3,
                "mm_skew_beta": 0,
                "mm_price_skew": 4,
                "mm_level_spacing": 5,
                "mm_spread_alpha": 0.75,
                "mm_backstop_quantity": 0,
                "mm_cancel_limit_delay": 50,
                "val_kappa": 1.67e-15,
                "val_vol": 1e-8,
                "computation_delay": 0,
            },
        },
        "framework": "torch",
        "seed": 1,
        
        # Per velocizzare, regola i parametri di training come serve a te
        "timesteps_per_iteration": 1000,
        "min_iter_time_s": 1, 

        # Parametri DQN
        "prioritized_replay": False,
        # Qui agganci la TUA classe di callback
        "callbacks": MyBaseCallbacksMkt,  

        "hiddens": [20, 50],
        "dueling": False,
        "double_q": False,
        "gamma": 0.99,
        "lr": 5e-6,
        "observation_filter": "MeanStdFilter",
        "buffer_size": 200000,
    },
)
