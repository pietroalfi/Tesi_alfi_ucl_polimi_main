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
import pickle

class MyCallbacks(DefaultCallbacks):

    # Modification for save buffer in checkpoints:

    def on_checkpoint_save(self, *, trainer, checkpoint_dir, **kwargs):
        # Salva il replay buffer
        replay_buffer_path = os.path.join(checkpoint_dir, "replay_buffer.pkl")
        buffer = trainer.workers.local_worker().replay_buffer
        with open(replay_buffer_path, "wb") as f:
            pickle.dump(buffer, f)
        print(f"Replay buffer saved in: {replay_buffer_path}")

    def on_checkpoint_restore(self, *, trainer, checkpoint_dir, **kwargs):
        # Ripristina il replay buffer
        replay_buffer_path = os.path.join(checkpoint_dir, "replay_buffer.pkl")
        if os.path.exists(replay_buffer_path):
            with open(replay_buffer_path, "rb") as f:
                buffer = pickle.load(f)
            trainer.workers.local_worker().replay_buffer = buffer
            print(f"✅ Replay buffer upload from: {replay_buffer_path}")


    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        episode.user_data = defaultdict(list)
        episode.user_data["q_values"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        policies: Dict[str, Policy],
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        policy = next(iter(policies.values()))  # Get the first (and likely only) policy
        episode.user_data["q_values"].append(policy.q_values)

        agent0_info = episode._agent_to_last_info["agent0"]
        for k, v in agent0_info.items():
            episode.user_data[k].append(v)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        q_values = np.array(episode.user_data["q_values"])
            
        episode.custom_metrics["q_value_mean"] = np.mean(q_values)
        episode.custom_metrics["q_value_max"] = np.max(q_values)
        episode.custom_metrics["q_value_min"] = np.min(q_values)

        # q_spread = np.max(q_values, axis=1) - np.min(q_values, axis=1)
        # episode.custom_metrics["q_value_spread_mean"] = np.mean(q_spread)

        q_probs = np.exp(q_values - np.max(q_values, axis=1, keepdims=True))
        q_probs = q_probs / np.sum(q_probs, axis=1, keepdims=True)
        entropy = -np.sum(q_probs * np.log(q_probs + 1e-10), axis=1)
        episode.custom_metrics["q_value_entropy_mean"] = np.mean(entropy)


        for metric in [
            "bid_volume", "ask_volume",
            "bid_exe_vol_agent", "ask_exe_vol_agent",
            "bid_volume_agent", "ask_volume_agent",
        ]:
            episode.custom_metrics[metric] = np.sum(episode.user_data[metric])

        episode.custom_metrics["avg_spread"] = np.mean(np.array(episode.user_data["ask"]) - np.array(episode.user_data["bid"]))
        
        for metric in [
            "cash_balance",
            "holdings"
        ]:
            episode.custom_metrics["final_"+metric] = episode.user_data[metric][-1]
            episode.custom_metrics[metric+"_vol"] = np.std(episode.user_data[metric])
            episode.custom_metrics[metric] = episode.user_data[metric]
        # return data
        starting_cash = 27_000
        final_ptf_value = episode.user_data["portfolio_value"][-1]
        episode.custom_metrics["episode_return"] = (final_ptf_value - starting_cash) / starting_cash

        # market return
        episode.custom_metrics["episode_market_return"] = (episode.user_data["mid"][-1] - episode.user_data["mid"][0])/episode.user_data["mid"][0]

        # excess market return 
        episode.custom_metrics["episode_excess_return"] = episode.custom_metrics["episode_return"] -  episode.custom_metrics["episode_market_return"]
        for i in range(0,3):
            episode.custom_metrics[f"d_signal_{i}"] = np.mean(episode.user_data[f"directional_signal_{i}"])
            episode.custom_metrics[f"d_signal_{i}_vol"] = np.std(episode.user_data[f"directional_signal_{i}"])

        # Maximum Drawdown
        portfolio_values = np.array(episode.user_data["portfolio_value"])
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (running_max - portfolio_values) / running_max
        max_drawdown = np.max(drawdowns)
        episode.custom_metrics["max_drawdown"] = max_drawdown

        # Mean Absolute Position
        episode.custom_metrics["mean_position"] = np.mean(episode.user_data["holdings"])
        episode.custom_metrics["position_vol_pct"] = np.std(episode.user_data["holdings"])/episode.custom_metrics["mean_position"] if (episode.custom_metrics["mean_position"]!=0) else 1
        episode.custom_metrics["mean_absolute_position"] = np.mean(np.abs(episode.user_data["holdings"]))
        episode.custom_metrics["mean_absolute_position_vol"] = np.std(np.abs(episode.user_data["holdings"]))

        action_counter = episode.user_data["action_counter"][-1]
        total_actions = sum(action_counter.values())
        for key, val in action_counter.items():
            episode.custom_metrics[f"pct_action_counter_{key}_{i}"] = val/total_actions

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        pass

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        pass


    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                          result: dict, **kwargs) -> None:
        pass
    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass
