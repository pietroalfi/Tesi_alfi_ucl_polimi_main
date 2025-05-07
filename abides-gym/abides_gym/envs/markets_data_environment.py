import importlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List
from itertools import takewhile

import gym
import numpy as np
import pandas as pd
from scipy.stats import dirichlet

import abides_markets.agents.utils as markets_agent_utils
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns, datetime_str_to_ns
from abides_core.generators import ConstantTimeGenerator

from .markets_environment import AbidesGymMarketsEnv


class SubGymGetInfoMkt(AbidesGymMarketsEnv):
    """
    Execution V0 environnement. It defines one of the ABIDES-Gym-markets environnement.
    This environment presents an example of the algorithmic optimal order-execution problem in the 
    context of market making. The current specification is based upon Nagy et al. (2023).


    Arguments:
        - background_config: the handcrafted agents configuration used for the environnement
        - mkt_close: time the market day ends
        - timestep_duration: how long between 2 wakes up of the gym experimental agent
        - starting_cash: cash of the agents at the beginning of the simulation
        - order_fixed_size: size of the order placed by the experimental gym agent
        - state_history_length: length of the raw state buffer
        - market_data_buffer_length: length of the market data buffer
        - debug_mode: arguments to change the info dictionnary (lighter version if performance is an issue)
        - background_config_extra_kvargs: dictionary of extra key value arguments passed to the background config builder function

    Execution Environment (custom):
        - Action Space:
            - LMT BUY order at bid, mid, or ask
            - LMT SELL order at bid, mid, or ask
            - Hold/wait
        - State Space:
            - time to close
            - cash balance
            - holdings
            - bid
            - ask
            - bid volume
            - ask volume
            - volume imbalance
            - momentum
            - directional signal
            - lagged variables
    """

    raw_state_pre_process = markets_agent_utils.ignore_buffers_decorator
    raw_state_to_state_pre_process = (
        markets_agent_utils.ignore_mkt_data_buffer_decorator
    )

    def __init__(
        self,
        background_config: Any = "fx_basic",
        timestep_duration: str = "1s",
        state_timesteps_back: int = 0,
        dirichlet_steps_ahead: int = 8,
        dirichlet_k: float =  0.00002,
        dirichlet_a_h: float = 100.0,
        dirichlet_a_l: float =  1.0,
        starting_cash: int = 2_000_000,
        max_holdings: int = 10,
        order_fixed_size: int = 1,
        state_history_length: int = 2,
        market_data_buffer_length: int = 5,
        w_direc: float = 1,
        reduction_factor: float = 0.99996,
        kappa: float = 1,
        debug_mode: bool = True,
        first_interval: str = "00:00:30",
        background_config_extra_kvargs: Dict[str, Any] = {},
    ) -> None:
        self.background_config: Any = importlib.import_module(
            "abides_markets.configs.{}".format(background_config), package=None
        )  
        self.timestep_duration: NanosecondTime = str_to_ns(timestep_duration)
        self.starting_cash: int = starting_cash
        self.max_holdings: int = max_holdings
        self.order_fixed_size: int = order_fixed_size
        self.state_history_length: int = state_history_length
        self.market_data_buffer_length: int = market_data_buffer_length
        self.debug_mode: bool = debug_mode
        self.first_interval: NanosecondTime = str_to_ns(first_interval)

        # Reward function params.
        self.w_direc: float = w_direc
        self.reduction_factor: float = reduction_factor
        self.kappa: float = kappa
        self.portfolio_mkt_value_t0: int = self.starting_cash
        self.state_timesteps_back: int = state_timesteps_back

        ##################
        # CHECK PROPERTIES

        assert (self.timestep_duration <= str_to_ns("06:30:00")) & (
            self.timestep_duration >= str_to_ns("00:00:00")
        ), "Select authorized timestep_duration"

        assert (type(self.starting_cash) == int) & (
            self.starting_cash >= 0
        ), "Select positive integer value for starting_cash"

        assert (type(self.order_fixed_size) == int) & (
            self.order_fixed_size >= 0
        ), "Select positive integer value for order_fixed_size"

        assert (type(self.state_history_length) == int) & (
            self.state_history_length >= 0
        ), "Select positive integer value for order_fixed_size"

        assert (type(self.market_data_buffer_length) == int) & (
            self.market_data_buffer_length >= 0
        ), "Select positive integer value for order_fixed_size"

        assert self.debug_mode in [
            True,
            False,
        ], "debug_mode needs to be True or False"

        background_config_args = {}
        background_config_args.update(background_config_extra_kvargs)
        super().__init__(
            background_config_pair=(
                self.background_config.build_config,
                background_config_args,
            ),
            wakeup_interval_generator=ConstantTimeGenerator(
                step_duration=self.timestep_duration
            ),
            starting_cash=self.starting_cash,
            state_buffer_length=self.state_history_length,
            market_data_buffer_length=self.market_data_buffer_length,
            first_interval=self.first_interval
        )

        # Action Space
        # do nothing
        self.num_actions: int = 1
        self.action_space: gym.Space = gym.spaces.Discrete(self.num_actions)

        # State Space
        # [T - t, cash balance]
        self.num_state_features: int = 2 + 6 * state_timesteps_back
        # construct state space "box"
        self.state_highs: np.ndarray = np.array(
            [
                np.finfo(np.float32).max,  # time_to_close
                np.finfo(np.float32).max,  # cash_balance
            ]
            + 6 * state_timesteps_back * [np.finfo(np.float32).max],  # state history vars
            dtype=np.float32,
        ).reshape(self.num_state_features, 1)

        self.state_lows: np.ndarray = np.array(
            [
                np.finfo(np.float32).min,  # time_to_close
                np.finfo(np.float32).min,  # cash_balance
            ]
            + 6 * state_timesteps_back * [0],  # state history vars
            dtype=np.float32,
        ).reshape(self.num_state_features, 1)

        self.observation_space: gym.Space = gym.spaces.Box(
            self.state_lows,
            self.state_highs,
            shape=(self.num_state_features, 1),
            dtype=np.float32,
        )


    def _map_action_space_to_ABIDES_SIMULATOR_SPACE(
        self, action: int
    ) -> List[Dict[str, Any]]:
        """
        utility function that maps open ai action definition (integers) to environnement API action definition (list of dictionaries)
        The action space ranges [0, 1, 2, 3, 4, 5, 6] where:
        - '0' LMT buy order_fixed_size at bid
        - '1' LMT buy order_fixed_size at midprice
        - '2' LMT buy order_fixed_size at ask
        - '3' LMT sell order_fixed_size at bid
        - '4' LMT sell order_fixed_size at midprice
        - '5' LMT sell order_fixed_size at ask
        - '6' DO NOTHING

        Arguments:
            - action: integer representation of the different actions

        Returns:
            - action_list: list of the corresponding series of action mapped into abides env apis
        """
        if action == 0:
            #[0,0]
                return [
                {"type": "CCL_ALL"}
                 ]
        else:
            raise ValueError(f"Action {action} is not part of the actions supported by the function.")
    

    @raw_state_to_state_pre_process
    def raw_state_to_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """
        Method that transforms a raw state into a state representation.
        State space:
        1. T - t
        2. C_t cash balance
        3. X_t stock inventory
        4. Price (bid/ask)
        5. Volume at best (bid/ask)
        6. Agent's posted volume (bid/ask)
        7. d_t directional signal (3x1)
        8. im_t volume imbalance (50 ts)
        9. M_t momentum

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - state: state representation defining the MDP for the execution v0 environnement
        """
        # 0) Preliminary
        bids = raw_state["parsed_mkt_data"]["bids"]
        asks = raw_state["parsed_mkt_data"]["asks"]
        last_transactions = raw_state["parsed_mkt_data"]["last_transaction"]

        # 1) Timing
        mkt_close = raw_state["internal_data"]["mkt_close"][-1]
        current_time = raw_state["internal_data"]["current_time"][-1]
        time_to_close = mkt_close - current_time

        # 2) Cash balance
        cash_balance = raw_state["internal_data"]["cash"][-1]

        # 3) Holdings
        holdings = raw_state["internal_data"]["holdings"][-1]

        # A) Calculate midprice
        mid_prices = [
            markets_agent_utils.get_mid_price(b, a, lt)
            for (b, a, lt) in zip(bids, asks, last_transactions)
        ]
        mid_price = mid_prices[-1]

        # 4) Spread & bid/ask prices
        best_bids = [
            bids[0][0] if len(bids) > 0 else mid
            for (bids, mid) in zip(bids, mid_prices)
        ]
        self.best_bid = best_bids[-1]
        best_asks = [
            asks[0][0] if len(asks) > 0 else mid
            for (asks, mid) in zip(asks, mid_prices)
        ]
        self.best_ask = best_asks[-1]
        # Utilised as state vars and prices for placing LMT orders
        self.bid = best_bids[-1]
        self.ask = best_asks[-1]
        self.mid_price = int(mid_price)
               
        print("Sta RUNNANDO")
        # log custom metrics to tracker
        self.custom_metrics_tracker.bid = self.bid
        self.custom_metrics_tracker.ask = self.ask
        self.custom_metrics_tracker.mid = self.mid_price
        self.custom_metrics_tracker.fundamental_value = self.kernel.oracle.observe_price(symbol="INTC", currentTime=current_time)
        self.custom_metrics_tracker.current_time = current_time

        # 8) Computed State
        hist_arrays_flat = np.concatenate(list(self.hist_arrays.values()))
        computed_state = np.concatenate([
            np.array([
                time_to_close,
                cash_balance,
            ])
        ])

        # Update historical values.
        if self.state_timesteps_back > 0:
            for key, arr in self.hist_arrays.items():
                arr = np.roll(arr, -1)
                arr[-1] = locals()[key.rsplit("_", 1)[0]]
                self.hist_arrays[key] = arr
        
        self.step_index += 1
        return computed_state.reshape(self.num_state_features, 1)

    @raw_state_pre_process
    def raw_state_to_reward(self, raw_state: Dict[str, Any]) -> float:
        """
        method that transforms a raw state into the reward obtained during the step

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: immediate reward computed at each step  for the execution v0 environnement
        """
        # Mark-to-Market Value:  M_t = C_t + X_tp_t^m
        # PnL Reward:            R_{t+1}^{PnL} = M_t - M_{t-1}
        # Directional Reward:    R_{t+1}^{dir} = k[-1,0,1]*d_tX_t
        # Total Reward:          r_{t+1} = w^{dir}*R_{t+1}^{dir} + (1 - w^{dir})*R_{t+1}^{PnL}

        holdings = raw_state["internal_data"]["holdings"]
        if self.step_index <= 1:
            self.portfolio_mkt_value_tm1 = self.portfolio_mkt_value_t0

        if holdings > 0:
            portfolio_mkt_value_t = holdings*self.bid + raw_state["internal_data"]["cash"]
        else:
            portfolio_mkt_value_t = holdings*self.ask + raw_state["internal_data"]["cash"]

        PnL_reward = (portfolio_mkt_value_t - self.portfolio_mkt_value_tm1) * self.kappa

        total_reward = PnL_reward

        return total_reward

    def raw_state_to_update_reward(self, raw_state: Dict[str, Any]) -> float:
        """
        method that transforms a raw state into the final step reward update (if needed)

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: update reward computed at the end of the episode for the execution v0 environnement
        """

        return 0

    @raw_state_pre_process
    def raw_state_to_done(self, raw_state: Dict[str, Any]) -> bool:
        """
        method that transforms a raw state into the flag if an episode is done

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - done: flag that describes if the episode is terminated or not  for the execution v0 environnement
        """
        # Episode can stop because market closes or because some condition is met
        # here the condition is market has closed or there is negative equity.
        
        current_time = raw_state["internal_data"]["current_time"]
        mkt_close = raw_state["internal_data"]["mkt_close"] - str_to_ns("5s")

        # If market closed or no equity return True else False.
        if current_time >= mkt_close or self.portfolio_mkt_value_tm1 < 0:
            return True
        else:
            return False

    @raw_state_pre_process
    def raw_state_to_info(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        method that transforms a raw state into an info dictionnary

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: info dictionnary computed at each step for the execution v0 environnement
        """
        # Agent cannot use this info for taking decision
        # only for debugging

        return asdict(self.custom_metrics_tracker)