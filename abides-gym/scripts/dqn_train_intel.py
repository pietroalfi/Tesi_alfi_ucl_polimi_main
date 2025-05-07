import ray
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
import wandb
import os
os.environ["WANDB_MODE"] = "offline"

#from abides_gym.envs.markets_fx_environment_basic import SubGymFxEnv_basic
from abides_gym.envs.markets_intc_environment_basic import SubGymIntcEnv_basic
from abides_gym.envs.markets_mm_custom_metrics_hf import MyBaseCallbacks_HF
from ray.tune.registry import register_env

# Registra l'ambiente personalizzato
register_env(
    "intc-basic-v0",
    lambda config: SubGymIntcEnv_basic(**config),
)

# Recupera la chiave API di WandB
api_key = wandb.api.api_key

# Arresta eventuali sessioni Ray precedenti
ray.shutdown()
ray.init()

# Configurazione e avvio dell'addestramento con DQN
tune.run(
    "DQN",  # Utilizza l'algoritmo DQN
    name="dqn_with_wandb_and_callbacks",  # Nome dell'esperimento
    resume=False,  # Non riprendere esperimenti interrotti
    stop={"training_iteration": 500},  
    checkpoint_at_end=True,  # Salva il checkpoint finale
    checkpoint_freq=10,  # Salva un checkpoint ogni 10 iterazioni
    config={
        # Environment
        "env": "intc-basic-v0",
        "env_config": {
            "background_config": "intel_basic",
            "timestep_duration": "0.2s",
            "state_timesteps_back": 0,
            "dirichlet_steps_ahead": 8,
            "dirichlet_k": 0.00005,
            "dirichlet_a_h": 5.0,
            "dirichlet_a_l": 1.0,
            "starting_cash": 200_000,
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
                "log_orders": False,
                "book_logging": False,
                "book_log_depth": 10,
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
                "mm_num_ticks": 5, #3
                "mm_min_order_size": 3,
                "mm_skew_beta": 0,
                "mm_price_skew": 4,
                "mm_level_spacing": 5, #2
                "mm_spread_alpha": 0.75,#0.85,
                "mm_backstop_quantity": 0,
                "mm_cancel_limit_delay": 50,
                "val_kappa": 1.67e-15,
                "val_vol": 100,#1e-8,
                "computation_delay": 0,
            },
        },
        # General
        "framework": "torch",
        "seed": 1,
        "num_workers": 0,
        "num_cpus_for_driver": 1,
        # Logging
        "timesteps_per_iteration": 1_000,
        "min_iter_time_s": 1, 
        # Model
        "prioritized_replay": False,  # Must set worker_side_prioritization to false as well
        "callbacks": MyBaseCallbacks_HF,  # Utilizza la tua classe di callback
        # Neural net
        "hiddens": [20, 50],
        "model": {
            "fcnet_hiddens": [64, 128, 64],
            "fcnet_activation": "tanh",
        },
        # Model
        "dueling": True,
        "double_q": True,
        "n_step": 10,
        "num_atoms": 51,
        "prioritized_replay": False,
        "target_network_update_freq": 5000,
        # Learning parameters
        "gamma": 0.99,
        # "lr_schedule": [[0,2e-5], [2.5e5, 5e-6]],
        "lr": 2e-4, #5e-6
        "observation_filter": "MeanStdFilter",  # Normalize observations
        "buffer_size": 200000,
    },
    callbacks=[
        WandbLoggerCallback(
            project="my_dqn_project",  # Nome del progetto WandB
            group="dqn_intel_experiments",  # Gruppo di esperimenti
            api_key=api_key,  # Chiave API di WandB
            log_config=False,  # Non loggare la configurazione completa
        )
    ],
)
