import ray
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
import wandb
from abides_gym.envs.markets_fx_environment_basic import SubGymFxEnv_basic
from abides_gym.envs.markets_mm_custom_metrics_hf import MyBaseCallbacks_HF
from ray.tune.registry import register_env

# Registra l'ambiente personalizzato
register_env(
    "fx-basic-v0",
    lambda config: SubGymFxEnv_basic(**config),
)

# Recupera la chiave API di WandB
api_key = wandb.api.api_key

# Arresta eventuali sessioni Ray precedenti
ray.shutdown()
ray.init(num_cpus=2)

# Configurazione e avvio dell'addestramento con DQN
tune.run(
    "DQN",  # Utilizza l'algoritmo DQN
    name="dqn_with_wandb_and_callbacks",  # Nome dell'esperimento
    resume=False,  # Non riprendere esperimenti interrotti
    stop={"training_iteration": 17},  
    checkpoint_at_end=True,  # Salva il checkpoint finale
    checkpoint_freq=10,  # Salva un checkpoint ogni 10 iterazioni
    config={
        # Environment
        "env": "fx-basic-v0",
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
            "w_direc": 1,
            "reduction_factor":  0.99996,
            "kappa": 5,
            "debug_mode": True,
            "first_interval": "00:00:30",
            "background_config_extra_kvargs": {
                "ticker": "EURUSD",
                #"period_starts": ['20240117 160100', '20240124 171100', '20240208 231800', '20240212 220500', '20240213 001000', '20240216 001200', '20240220 172000', '20240226 155000', '20240305 170700', '20240311 161400', '20240312 160400', '20240314 224500', '20240317 231000', '20240318 160400', '20240320 223900', '20240321 164200', '20240326 161200', '20240327 231300', '20240329 002800', '20240329 120000', '20240329 130600', '20240401 155400', '20240424 170200', '20240425 160100', '20240425 171800', '20240507 160200', '20240512 175800', '20240521 004100', '20240521 171800', '20240605 171200'],
                "fundamental_file_path": "/home/pietro/ucl-thesis-main/ucl-thesis-main/data/EURUSD_2024_01_midpoint_1000.pkl",
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
                "val_lambda_a": 7e-11, #1e-11
                "num_mm": 2,
                "mm_wake_up_freq": "0.1s",
                "mm_window_size": "adaptive",
                "mm_pov": 0.08,
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
        # General
        "framework": "torch",
        "seed": 1,
        "num_workers": 0,
        "num_cpus_per_worker": 4,
        ""
        # Logging
        "timesteps_per_iteration": 1_000,
        "min_iter_time_s": 1, 
        # Model
        "prioritized_replay": False,  # Must set worker_side_prioritization to false as well
        "callbacks": MyBaseCallbacks_HF,  # Utilizza la tua classe di callback
        # Neural net
        "hiddens": [20, 50],
        # Model
        "dueling": True,
        "double_q": True,
        #"num_atoms": 51,
        "prioritized_replay": False,
        "target_network_update_freq": 5000,
        # Learning parameters
        "gamma": 0.99,
        # "lr_schedule": [[0,2e-5], [2.5e5, 5e-6]],
        "lr": 5e-6,
        "observation_filter": "MeanStdFilter",  # Normalize observations
        "buffer_size": 200000,
    },
    callbacks=[
        WandbLoggerCallback(
            project="my_dqn_project",  # Nome del progetto WandB
            group="dqn_fx_experiments",  # Gruppo di esperimenti
            api_key=api_key,  # Chiave API di WandB
            log_config=False,  # Non loggare la configurazione completa
        )
    ],
)
