# RMSC-3 (Reference Market Simulation Configuration):
# - 1     Exchange Agent
# - 1     POV Market Maker Agent
# - 100   Value Agents
# - 25    Momentum Agents
# - 5000  Noise Agents
# - 1     (Optional) POV Execution agent


import numpy as np
import pandas as pd
import sys
import datetime as dt


from abides_markets.oracles.ExternalFileOracle import ExternalFileOracle

from abides_core.utils import str_to_ns, datetime_str_to_ns, get_wake_time
from abides_markets.agents import (
    ExchangeAgent,
    NoiseAgent,
    ValueAgent,
    AdaptiveMarketMakerAgent,
    MomentumAgent,
)
from abides_markets.utils import generate_latency_model
from abides_markets.models import OrderSizeModel


########################################################################################################################
############################################### GENERAL CONFIG #########################################################


def build_config(
    ticker="EURUSD",
    historical_date="20240102",
    start_time="00:05:00",
    end_time="00:30:00",
    fundamental_file_path="data/EURUSD_midpoint.pkl",  # <-----------
    exchange_log_orders=True,
    log_orders=True,
    book_logging=True,
    book_log_depth=10,
    #   seed=int(NanosecondTime.now().timestamp() * 1000000) % (2 ** 32 - 1),
    seed=1,
    stdout_log_level="INFO",
    ##
    num_momentum_agents=25,
    num_noise_agents=5000,
    num_value_agents=100,
    ## exec agent
    execution_agents=True,
    execution_pov=0.1,
    # 4) Market Maker Agents
    # each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)
    mm_window_size="adaptive",
    mm_pov=0.025,
    mm_num_ticks=10,
    mm_wake_up_freq="60S",
    mm_min_order_size=1,
    mm_skew_beta=0,
    mm_price_skew=4,
    mm_level_spacing=5,
    mm_spread_alpha=0.75,
    mm_backstop_quantity=0,
    mm_cancel_limit_delay=50,  # 50 nanoseconds
    ##fundamental/oracle
    fund_r_bar=100_000,
    fund_kappa=1.67e-16,
    fund_sigma_s=0,
    fund_vol=1e-3,  # Volatility of fundamental time series (std).
    fund_megashock_lambda_a=2.77778e-18,
    fund_megashock_mean=1000,
    fund_megashock_var=50_000,
    ##value agent
    val_r_bar=100_000,
    val_kappa=1.67e-15,
    val_vol=1e-8,
    val_lambda_a=7e-11,
):

    fund_sigma_n = fund_r_bar / 10
    val_sigma_n = val_r_bar / 10
    symbol = ticker

    ##setting numpy seed
    np.random.seed(seed)

    ########################################################################################################################
    ############################################### AGENTS CONFIG ##########################################################

    # Historical date to simulate.
    historical_date = datetime_str_to_ns(historical_date)
    mkt_open = historical_date + str_to_ns(start_time)
    mkt_close = historical_date + str_to_ns(end_time)
    noise_mkt_open = mkt_open - str_to_ns("00:01:00")
    agent_count, agents, agent_types = 0, [], []

    # Hyperparameters
    starting_cash = 10000000  # Cash in this simulator is always in CENTS.

    # Oracle
    symbols = {
        symbol : {
            'fundamental_file_path': fundamental_file_path,
            'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))
        }
    }
    oracle = ExternalFileOracle(symbols)

    # Params for noise agents.
    r_bar = oracle.fundamentals[symbol].values[0]
    sigma_n = r_bar / 100
    kappa = 1.67e-15
    lambda_a = 1e-12
    ORDER_SIZE_MODEL = OrderSizeModel()  # Order size model

    # 1) Exchange Agent
    agents.extend(
        [
            ExchangeAgent(
                id=0,
                name="EXCHANGE_AGENT",
                type="ExchangeAgent",
                mkt_open=mkt_open,
                mkt_close=mkt_close,
                symbols=[symbol],
                book_logging=book_logging,
                book_log_depth=book_log_depth,
                log_orders=exchange_log_orders,
                pipeline_delay=0,
                computation_delay=0,
                stream_history=25_000,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
            )
        ]
    )
    agent_types.extend("ExchangeAgent")
    agent_count += 1

    # 2) Noise Agents
    num_noise = 200

    agents.extend(
            [
                NoiseAgent(
                    id=j,
                    name="NoiseAgent {}".format(j),
                    type="NoiseAgent",
                    symbol=symbol,
                    starting_cash=starting_cash,
                    wakeup_time=get_wake_time(noise_mkt_open, mkt_close),
                    log_orders=log_orders,
                    random_state=np.random.RandomState(
                        seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
            )
            for j in range(agent_count, agent_count + num_noise)
        ]
    )
    agent_count += num_noise
    agent_types.extend(['NoiseAgent'])

    # 3) Value Agents
    num_value = 100
    agents.extend(
        [
            ValueAgent(
                id=j,
                name="Value Agent {}".format(j),
                type="ValueAgent",
                symbol=ticker,
                starting_cash=starting_cash,
                sigma_n=sigma_n,
                r_bar=r_bar,
                kappa=kappa,
                lambda_a=lambda_a,
                log_orders=log_orders,
                order_size_model=ORDER_SIZE_MODEL,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
            )
            for j in range(agent_count, agent_count + num_value_agents)
        ]
    )
    agent_count += num_value
    agent_types.extend(['ValueAgent'])


    ########################################################################################################################
    ########################################### KERNEL AND OTHER CONFIG ####################################################

    # extract kernel seed here to reproduce the state of random generator in old version
    random_state_kernel = np.random.RandomState(
        seed=np.random.randint(low=0, high=2**32, dtype="uint64")
    )

    kernelStartTime = historical_date
    kernelStopTime = mkt_close + str_to_ns('00:01:00')

    latency_model = generate_latency_model(agent_count)
    default_computation_delay = 50  # 50 nanoseconds

    return {
        "seed": seed,
        "start_time": kernelStartTime,
        "stop_time": kernelStopTime,
        "agents": agents,
        "agent_latency_model": latency_model,
        "default_computation_delay": default_computation_delay,
        "custom_properties": {"oracle": oracle},
        "random_state_kernel": random_state_kernel,
        "stdout_log_level": stdout_log_level,
    }

    
