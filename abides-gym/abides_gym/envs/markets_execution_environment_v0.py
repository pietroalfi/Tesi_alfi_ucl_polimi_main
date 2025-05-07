import importlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import gym
import numpy as np
import pandas as pd
from pprint import pp
from scipy.stats import dirichlet

import abides_markets.agents.utils as markets_agent_utils
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns, datetime_str_to_ns
from abides_core.generators import ConstantTimeGenerator

from .markets_environment import AbidesGymMarketsEnv


class SubGymMarketsExecutionEnv_v0(AbidesGymMarketsEnv):
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
        background_config: Any = "FX_test_execution",
        timestep_duration: str = "0.2s",
        state_timesteps_back: int = 0,
        dirichlet_steps_ahead: int = 8,
        dirichlet_k: float =  0.0002,
        dirichlet_a_h: float = 5.0,
        dirichlet_a_l: float =  1.0,
        w_direc: float = 1,
        reduction_factor: float = 0.9999,
        kappa: float = 1.0,
        starting_cash: int = 2_000_000,
        order_fixed_size: int = 1,
        state_history_length: int = 2,
        market_data_buffer_length: int = 5,
        debug_mode: bool = False,
        first_interval: str = "00:00:30",
        background_config_extra_kvargs: Dict[str, Any] = {},
    ) -> None:
        self.background_config: Any = importlib.import_module(
            "abides_markets.configs.{}".format(background_config), package=None
        )  
        self.timestep_duration: NanosecondTime = str_to_ns(timestep_duration)
        self.starting_cash: int = starting_cash
        self.order_fixed_size: int = order_fixed_size
        self.state_history_length: int = state_history_length
        self.market_data_buffer_length: int = market_data_buffer_length
        self.debug_mode: bool = debug_mode
        self.first_interval: NanosecondTime = str_to_ns(first_interval)
        
        # Initialise params.
        self.portfolio_mkt_value_t0: int = self.starting_cash
        self.state_timesteps_back: int = state_timesteps_back

        # Dirichlet alpha signal params.
        self.dhlt_steps_ahead: int = dirichlet_steps_ahead
        self.dhlt_params: List[float] = [dirichlet_a_h, dirichlet_a_l, dirichlet_k, 0.3]

        # Reward function params.
        self.w_direc: float = w_direc
        self.reduction_factor: float = reduction_factor
        self.kappa: float = kappa

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
        # Limit order @ (bid, midpoint, ask) x (buy, sell) + do nothing.
        self.num_actions: int = 7
        self.action_space: gym.Space = gym.spaces.Discrete(self.num_actions)

        # State Space
        # [T - t, cash balance, stock inventory, price (bid/ask), volume at best (bid/ask),
        #  agent's posted volume (bid/ask), directional signal (3x1)]
        self.num_state_features: int = 14 + 6*state_timesteps_back
        # construct state space "box"
        self.state_highs: np.ndarray = np.array(
            [
                np.iinfo(np.int64).max,  # time_to_close NOTE: could we switch to 32-bit?
                np.iinfo(np.int64).max,  # cash_balance
                np.iinfo(np.int64).max,  # holdings
                np.iinfo(np.int64).max,  # bid (price)
                np.iinfo(np.int64).max,  # ask (price)
                np.iinfo(np.int64).max,  # bid_volume
                np.iinfo(np.int64).max,  # ask_volume
                np.iinfo(np.int64).max,  # bid_volume_agent
                np.iinfo(np.int64).max,  # ask_volume_agent
                np.iinfo(np.int64).max,  # volume_imbalance
                np.iinfo(np.int64).max,  # momentum
            ]
            + 3*[np.finfo(np.float32).max]  # 3 * directional probability
            + 6*state_timesteps_back*[np.finfo(np.float32).max], # state history vars
            dtype=np.float32,
        ).reshape(self.num_state_features, 1)

        self.state_lows: np.ndarray = np.array(
            [
                0,  # time_to_close
                np.iinfo(np.int64).min,  # cash_balance
                np.iinfo(np.int64).min,  # holdings
                0,  # bid (price)
                0,  # ask (price)
                0,  # bid_volume
                0,  # ask_volume
                0,  # bid_volume_agent
                0,  # ask_volume_agent
                np.iinfo(np.int64).min,  # volume_imbalance
                np.iinfo(np.int64).min  # momentum
            ]
            + 3*[np.finfo(np.float32).min]
            + 6*state_timesteps_back*[0],
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
        holdings = self.state[2,0] if self.state is not None else 0
        buy_actions, sell_actions = [0,1,2], [3,4,5]
        
        if holdings > 9 and action in buy_actions:
            return []

        elif holdings < -9 and action in sell_actions:
            return []

        self.custom_metrics_tracker.action_counter[
            f"action_{action}"
        ] += 1  # increase counter
        if action == 0:
            return [
                {"type": "CCL_ALL"},  # CCL_ALL = cancel all
                {
                    "type": "LMT",
                    "direction": "BUY",
                    "size": self.order_fixed_size,
                    "limit_price": self.bid,
                },
            ]

        elif action == 1:
            return [
                {"type": "CCL_ALL"},
                {
                    "type": "LMT",
                    "direction": "BUY",
                    "size": self.order_fixed_size,
                    "limit_price": self.mid_price,
                },
            ]
        elif action == 2:
            return [                
                {"type": "CCL_ALL"},
                {
                    "type": "LMT",
                    "direction": "BUY",
                    "size": self.order_fixed_size,
                    "limit_price": self.ask,
                 },
            ]
        elif action == 3:
            return [                
                {"type": "CCL_ALL"},
                {
                    "type": "LMT",
                    "direction": "SELL",
                    "size": self.order_fixed_size,
                    "limit_price": self.bid,
                 },
            ]
        elif action == 4:
            return [                
                {"type": "CCL_ALL"},
                {
                    "type": "LMT",
                    "direction": "SELL",
                    "size": self.order_fixed_size,
                    "limit_price": self.mid_price,
                 },
            ]
        elif action == 5:
            return [                
                {"type": "CCL_ALL"},
                {
                    "type": "LMT",
                    "direction": "SELL",
                    "size": self.order_fixed_size,
                    "limit_price": self.ask,
                    },
            ]
        
        elif action == 6:
            return []
        
        else:
            raise ValueError(
                f"Action {action} is not part of the actions supported by the function."
            )

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
        best_asks = [
            asks[0][0] if len(asks) > 0 else mid
            for (asks, mid) in zip(asks, mid_prices)
        ]

        spreads = np.array(best_asks) - np.array(best_bids)
        spread = spreads[-1]
        
        # Utilised as state vars and prices for placing LMT orders
        self.bid = bid = best_bids[-1]
        self.ask = ask = best_asks[-1]
        self.mid_price = int(mid_price)

        # 6) Volume at best
        # 6a) Total volume available on each side of book.
        bid_volume = raw_state["parsed_volume_data"]["bid_volume"][-1]
        ask_volume = raw_state["parsed_volume_data"]["ask_volume"][-1]
        
        # 6b) Volume posted by agent.
        # Orders in the state is an ever-increasing dictionary and thus need to ensure 
        # not doublely checking whether orders have been executed.
        orders = list(raw_state["internal_data"]["order_status"][-1].values())[self.farthest_executed_idx:]
        active_orders = [(idx, order["active_qty"], order["order"].side.is_bid()) for idx, order in enumerate(orders) if order["status"] == "active"]
        if active_orders:
            self.farthest_executed_idx = active_orders[0][0]
        else:
            self.farthest_executed_idx = -1

        bid_volume_agent = np.sum([order[1] for order in active_orders if order[2]])
        ask_volume_agent = np.sum([order[1] for order in active_orders if not order[2]])

        # 7) Directional signal
        # d_t = phi*d_{t-1} + (1 - phi)epsilon_t
        # epslion_t derives from a dirichlet with parameter vector alpha.
        if self.step_index == -1:
            self.direc_signal_t = np.array([1/3, 1/3, 1/3])
        # Determine parameters.
        future_times = [current_time + x*self.timestep_duration for x in range(self.dhlt_steps_ahead)]
        future_fundamental_prices = [self.kernel.oracle.observe_price(symbol="EURUSD", currentTime=ft) for ft in future_times]
        avg_future_price = np.mean(future_fundamental_prices)

        avg_future_ret = (avg_future_price - mid_price)/mid_price
        self.avg_future_ret = avg_future_ret

        a_H, a_L, k, phi = self.dhlt_params
        if avg_future_ret < -k:
            alpha = [a_H, a_L, a_L]
        elif -k <= avg_future_ret < k:
            alpha = [a_L, a_H, a_L]
        else:
            alpha = [a_L, a_L, a_H]
    
        # Sample from Dirichlet distribution
        epsilon_t = dirichlet.rvs(alpha)[0]
        
        # Update directional signal
        self.direc_signal_t = phi * self.direc_signal_t + (1 - phi) * epsilon_t

        self.volume_hist = np.roll(self.volume_hist, -1)
        self.volume_hist[-1] = bid_volume - ask_volume
        volume_imbalance = np.sum(self.volume_hist)/np.sum(np.abs(self.volume_hist))
        volume_imbalance = np.nan_to_num(volume_imbalance, nan=0.0)  # if there are 50 steps of no vol --> num/0 = nan

        self.price_hist_5 = np.roll(self.price_hist_5, -1)
        self.price_hist_30 = np.roll(self.price_hist_30, -1)
        self.price_hist_5[-1] = self.mid_price
        self.price_hist_30[-1] = self.mid_price

        ewma_5 = pd.Series(self.price_hist_5[self.price_hist_5 != 0]).ewm(span=5, adjust=False).mean()
        ewma_30 = pd.Series(self.price_hist_30[self.price_hist_30 != 0]).ewm(span=30, adjust=False).mean()
 
        momentum = (ewma_5.iloc[-1] - ewma_30.iloc[-1])
        momentum = np.nan_to_num(momentum, nan=0.0)

        # log custom metrics to tracker
        self.custom_metrics_tracker.cash_balance = cash_balance
        self.custom_metrics_tracker.holdings = holdings
        self.custom_metrics_tracker.bid = self.bid
        self.custom_metrics_tracker.ask = self.ask
        self.custom_metrics_tracker.bid_volume = bid_volume
        self.custom_metrics_tracker.ask_volume = ask_volume
        self.custom_metrics_tracker.bid_volume_agent = bid_volume_agent
        self.custom_metrics_tracker.ask_volume_agent = ask_volume_agent
        self.custom_metrics_tracker.directional_signal_0 = self.direc_signal_t[0]
        self.custom_metrics_tracker.directional_signal_1 = self.direc_signal_t[1]
        self.custom_metrics_tracker.directional_signal_2 = self.direc_signal_t[2]

        #print(self.kernel.oracle.observe_price(symbol="EURUSD", currentTime=current_time))
        #print(self.bid)

        # 8) Computed State
        hist_arrays_flat = np.concatenate(list(self.hist_arrays.values()))
        computed_state = np.concatenate([
            np.array([
                time_to_close,
                cash_balance,
                holdings,
                self.bid,
                self.ask,
                bid_volume,
                ask_volume,
                bid_volume_agent,
                ask_volume_agent,
                volume_imbalance,
                momentum
            ]),
            self.direc_signal_t,
            hist_arrays_flat
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
        # PnL Reward:            R_{t+1}^{PnL} = ln(M_t) - ln(M_{t-1})
        # Directional Reward:    R_{t+1}^{dir} = k[-1,0,1]*d_tX_t
        # Total Reward:          r_{t+1} = w^{dir}*R_{t+1}^{dir} + (1 - w^{dir})*R_{t+1}^{PnL}

        holdings = raw_state["internal_data"]["holdings"]
        if self.step_index <= 1:
            self.portfolio_mkt_value_tm1 = self.portfolio_mkt_value_t0

        currentTime = raw_state["internal_data"]["current_time"]
        fundamental_value = self.kernel.oracle.observe_price("EURUSD", currentTime)
        modified_price = 0.85*(self.mid_price - fundamental_value) + fundamental_value
        portfolio_mkt_value_t = holdings*modified_price + raw_state["internal_data"]["cash"]

        PnL_reward = portfolio_mkt_value_t - self.portfolio_mkt_value_tm1
        self.portfolio_mkt_value_tm1 = portfolio_mkt_value_t

        directional_reward = self.kappa*np.sum(self.direc_signal_t*[-1.0,0,1.0])*raw_state["internal_data"]["holdings"]
        #print("PnL_reward: ", PnL_reward, "Directional_reward: ", directional_reward)
        #print("Port_Val_t: ", portfolio_mkt_value_t, "Port_Val_tm1", self.portfolio_mkt_value_tm1)
        #print("Bid/ask: ", self.bid, self.ask)
        total_reward = self.w_direc*directional_reward + (1 - self.w_direc)*PnL_reward
        #print("Total_reward: ", total_reward)
        # print("")

        self.pnl, self.direc = (1 - self.w_direc)*PnL_reward, self.w_direc*directional_reward

        self.w_direc *= self.reduction_factor
        
        self.custom_metrics_tracker.portfolio_value = portfolio_mkt_value_t
        self.custom_metrics_tracker.pnl_reward = PnL_reward
        self.custom_metrics_tracker.directional_reward = directional_reward
        self.custom_metrics_tracker.total_reward = total_reward

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

        # 1) Last Known Market Transaction Price
        last_transaction = raw_state["parsed_mkt_data"]["last_transaction"]

        # 2) Last Known best bid
        bids = raw_state["parsed_mkt_data"]["bids"]
        best_bid = bids[0][0] if len(bids) > 0 else last_transaction

        # 3) Last Known best ask
        asks = raw_state["parsed_mkt_data"]["asks"]
        best_ask = asks[0][0] if len(asks) > 0 else last_transaction

        # 4) Current Time
        current_time = raw_state["internal_data"]["current_time"]

        # 5) Holdings
        holdings = raw_state["internal_data"]["holdings"]

        # 6) PnL
        market_value_portfolio = raw_state["internal_data"]["holdings"]*self.bid + raw_state["internal_data"]["cash"]
        PnL = market_value_portfolio - self.starting_cash

        if self.debug_mode == True:
            return {
                "last_transaction": last_transaction,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "current_time": current_time,
                "holdings": holdings,
                "pnl": PnL,
            }
        else:
            return asdict(self.custom_metrics_tracker)