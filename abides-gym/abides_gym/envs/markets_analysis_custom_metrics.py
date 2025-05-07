import os
import pickle
from collections import defaultdict
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.env import BaseEnv
from typing import Dict

class PartialMetricsStore:
    """
    Singleton for collecting raw values from the custom tracker.
    
    - Stores step-wise data in memory.
    - Appends all steps at once at the end of the episode to reduce file I/O.
    """
    pkl_file_path = "training_data.pkl"
    data = []  # List of dictionaries, each representing raw values for one step.

    @classmethod
    def reset(cls):
        """Reset in-memory data (the pickle file remains unchanged)."""
        print("Reset in-memory raw metrics store")
        cls.data = []

    @classmethod
    def update(cls, custom_tracker: dict):
        """Store step data in memory instead of writing each step to disk immediately."""
        cls.data.append(custom_tracker)

    @classmethod
    def save_to_disk(cls):
        """Append all stored step data to the pickle file in one operation."""
        if cls.data:
            with open(cls.pkl_file_path, "ab") as f:
                pickle.dump(cls.data, f)  # Save the entire list at once
            cls.reset()  # Clear memory after saving


class MyBaseCallbacksMkt(DefaultCallbacks):
    """
    Callbacks for the market maker environment that collect raw metrics
    directly from the custom tracker. Instead of saving each step separately,
    it batches and writes at the end of the episode or training iteration.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        print("Episode start: initializing episode data")
        episode.user_data = defaultdict(list)
        episode.hist_data["custom_tracker"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        # Get the raw custom tracker dictionary
        agent0_info = episode._agent_to_last_info["agent0"]
        episode.user_data["custom_tracker"].append(agent0_info)
        # Store it in memory instead of writing to disk immediately
        PartialMetricsStore.update(agent0_info)

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
        print("Episode end: saving collected step data to disk.")
        # Save all collected step data for this episode at once
        PartialMetricsStore.save_to_disk()

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        print("Training result callback: saving all pending raw custom metrics.")
        # Ensure any remaining data is saved at the end of a training iteration
        PartialMetricsStore.save_to_disk()
        result["custom_metrics"]["raw_steps"] = len(PartialMetricsStore.data)
