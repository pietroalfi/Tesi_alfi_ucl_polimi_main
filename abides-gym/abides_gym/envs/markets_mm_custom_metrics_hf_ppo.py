from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.env import BaseEnv

from typing import Dict, Tuple
from collections import defaultdict
import numpy as np
import os
import csv

class PartialMetricsStore:
    """Singleton per metriche parziali tra istanze di callback."""
    data = {"reward": [], "spread": [], "holdings": [], "value": [], "action_spread": [], "mid_price": [], "current_time": []}
    file_initialized = False  # Controlla se il file è già stato inizializzato

    @classmethod
    def reset(cls):
        print("Reset delle metriche parziali")
        cls.data = {"reward": [], "spread": [], "holdings": [], "value": [], "action_spread": [], "mid_price": [], "current_time": []}

    @classmethod
    def update(cls, reward, spread, holdings, value, action_spread, mid_price, current_time):
        cls.data["reward"].append(reward)
        cls.data["spread"].append(spread)
        cls.data["holdings"].append(holdings)
        cls.data["value"].append(value)
        cls.data["action_spread"].append(action_spread)
        cls.data["mid_price"].append(mid_price)
        cls.data["current_time"].append(current_time)

    @classmethod
    def get_averages(cls, output_file='training_data.csv'):
        print(f"Tentativo di salvataggio nel file CSV: {os.path.abspath(output_file)}")
        # Controlla se il file è già stato inizializzato
        if not cls.file_initialized:
            if not os.path.exists(output_file):
                print(f"File non trovato. Creazione del file: {output_file}")
                with open(output_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["current_time", "mid_price"])  # Intestazioni del file
            else:
                print(f"File trovato: {output_file}")
            cls.file_initialized = True

        # Salva i dati di mid_price e current_time nel file CSV
        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            rows = zip(cls.data["current_time"], cls.data["mid_price"])
            writer.writerows(rows)

        # Calcola le metriche
        averages = {
            "reward": np.mean(cls.data["reward"]) if cls.data["reward"] else 0,
            "spread_avg_partial": np.mean(cls.data["spread"]) if cls.data["spread"] else 0,
            "holdings_avg_partial": np.mean(cls.data["holdings"]) if cls.data["holdings"] else 0,
            "cash_performance_partial": (cls.data["value"][-1] - 200_000) if cls.data["value"] else 0,
            "Posted Spread Partial": np.mean(cls.data["action_spread"]) if cls.data["action_spread"] else 0,
            "Mean Absolute Position": np.mean(np.abs(cls.data["holdings"])) if cls.data["holdings"] else 0,
        }
        cls.reset()
        print(f"File CSV aggiornato: {output_file}")
        return averages


class MyBaseCallbacks_HF_ppo(DefaultCallbacks):
    """
    Class that defines callbacks for il market maker environments
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.RELEVANT_KEYS = [
            "spread",
            "holdings",
            "action",
            "book imbalance",
            "value"
        ]
        self.global_hist_data = defaultdict(list)

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        print("Inizio dell'episodio: creazione delle metriche iniziali")
        assert episode.length == 0, "ERROR: `on_episode_start()` should be called right after env reset!"
        episode.user_data = defaultdict(list)
        for key in self.RELEVANT_KEYS:
            episode.user_data[key] = []
        episode.user_data["rewards"] = []
        episode.hist_data["daily_net_cash_performance"] = []
        episode.hist_data["action distribution"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        assert episode.length > 0, "ERROR: `on_episode_step()` should not be called right after env reset!"
        agent0_info = episode._agent_to_last_info["agent0"]
        print(agent0_info)
        for k, v in agent0_info.items():
            if k in self.RELEVANT_KEYS:
                episode.user_data[k].append(v)
        episode.user_data["rewards"].append(agent0_info.get("reward", 0))

        reward = agent0_info.get("reward", 0)
        spread = agent0_info.get("spread", 0)
        holdings = agent0_info.get("holdings", 0)
        value = agent0_info.get("value", 0)
        action = agent0_info.get("action", 0)
        if isinstance(action, tuple) and len(action) == 2:
            action_spread = action[1] - action[0]
        else:
            action_spread = 0
        mid_price = agent0_info.get("mid_price", 0)
        current_time = agent0_info.get("current_time", 0)
        PartialMetricsStore.update(reward, spread, holdings, value, action_spread, mid_price, current_time)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        print("Fine dell'episodio: calcolo delle metriche finali")
        for key in ["spread", "holdings"]:
            episode.custom_metrics[f"{key}_avg"] = np.mean(episode.user_data[key])
        episode.custom_metrics["Mean Absolute Position"] = np.mean(np.abs(episode.user_data["holdings"]))
        episode.custom_metrics["Daily Net Cash Performance"] = episode.user_data["value"][-1] - 2_000_000

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        print("Risultato del training: calcolo delle metriche globali")
        averages = PartialMetricsStore.get_averages(output_file='training_data.csv')
        print(f"Averages calcolati: {averages}")
        result["custom_metrics"].update(averages)
        PartialMetricsStore.reset()
