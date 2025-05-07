import ray
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
import wandb

from abides_gym.envs.markets_fx_environment_basic_ppo import SubGymFxEnv_basic_ppo
from abides_gym.envs.markets_mm_custom_metrics_hf_ppo import MyBaseCallbacks_HF_ppo
from ray.tune.registry import register_env

# 1) Registra l'ambiente personalizzato
register_env(
    "fx-basic-v0-ppo",
    lambda config: SubGymFxEnv_basic_ppo(**config),
)

# 2) Recupera la chiave API di WandB
api_key = wandb.api.api_key

# 3) Arresta eventuali sessioni Ray precedenti e inizializza Ray
ray.shutdown()
ray.init(num_cpus=2)

# 4) Configurazione e avvio dell'addestramento con PPO
tune.run(
    "PPO",  
    name="ppo_with_wandb_and_callbacks",
    resume=False,
    stop={"training_iteration": 500},
    checkpoint_at_end=True,
    checkpoint_freq=10,
    config={
        # -- ENVIRONMENT --
        "env": "fx-basic-v0-ppo",
        "env_config": {
            "background_config": "fx_basic",
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
            "w_direc": 1.0,
            "reduction_factor": 0.99996,
            "kappa": 10,
            "debug_mode": True,
            "first_interval": "00:00:30",
            "background_config_extra_kvargs": {
                "ticker": "EURUSD",
                "fundamental_file_path": "/home/pietro/ucl-thesis-main/ucl-thesis-main/data/EURUSD_2024_01_midpoint_1000.pkl",
                "exchange_log_orders": False,
                "log_orders": False,
                "book_logging": False,
                "book_log_depth": 10,
                "starting_cash": 2_000_000,
                "seed": 1,
                "num_noise_agents": 5000,
                "num_momentum_agents": 40,
                "m_wake_up_freq": "0.5s",
                "num_value_agents": 200,
                "val_wake_up_freq": "60s",
                "val_lambda_a": 7e-11,
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
        # -- PARAMETRI GENERALI --
        "framework": "torch",
        "seed": 1,
        "observation_filter": "MeanStdFilter",  # normalizza le osservazioni
        
        # -- PARAMETRI PPO --
        "num_sgd_iter": 10,          # Quanti epoch di SGD a ogni iter
        "sgd_minibatch_size": 128,   # Grandezza dei mini-batch
        "train_batch_size": 4000,    # Grandezza del batch completo per iter
        "clip_param": 0.2,           # PPO clipping
        "lr": 3e-5,                  # learning rate di partenza
        "gamma": 0.99,               # discount factor
        "lambda": 0.95,              # GAE (Generalized Advantage Estimation)
        "vf_clip_param": 200.0,      # Clipping sul value function loss
        "entropy_coeff": 0.01,       # Aggiunge esplorazione (reg. entropia)

        # -- MODELLO (rete neurale) --
        "model": {
            "fcnet_hiddens": [128, 128],   # Numero neuroni
            "fcnet_activation": "tanh",
        },

        # -- RISORSE --
        "num_workers": 0,  # Usa un solo worker per evitare problemi di risorse
        "num_gpus": 0,     # Esplicitamente zero GPU per evitare errori

        # -- LOGGING & CALLBACK --
        "callbacks": MyBaseCallbacks_HF_ppo,
        "timesteps_per_iteration": 1000,
        "min_iter_time_s": 1,
    },
    callbacks=[
        WandbLoggerCallback(
            project="my_ppo_project",  
            group="ppo_fx_experiments", 
            api_key=api_key,             
            log_config=False,
        )
    ],
)
