# Derived from the RMSC-4 config:
# - 1     Exchange Agent
# - 2     Adaptive Market Maker Agents
# - 100   Value Agents
# - 40    Momentum Agents
# - 5000  Noise Agents


import numpy as np
import pandas as pd
import sys
from datetime import datetime


from abides_markets.oracles.ExternalFileOracle import ExternalFileOracle
from abides_markets.oracles import SparseMeanRevertingOracle

from abides_core.utils import str_to_ns, datetime_str_to_ns, get_wake_time, random_weekday
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
    period_starts= ["20240104 090000"], #"20240101 170012"
    fundamental_file_path="/cs/student/projects3/cf/2023/fdobber/ucl-thesis/data/EURUSD_2024_01_midpoint_1000.pkl",
    exchange_log_orders=False,
    log_orders=False,
    book_logging=False,
    book_log_depth=10,
    starting_cash=2_000_000,
    seed=1,
    stdout_log_level="INFO",
    num_noise_agents=5000,
    num_momentum_agents=40,
    m_wake_up_freq="0.5s",
    num_value_agents=200,
    val_wake_up_freq="45s",
    num_mm=2,
    mm_wake_up_freq="0.1s",
    # 4) Market Maker Agents
    # each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)
    mm_window_size="adaptive",
    mm_pov=0.025,
    mm_num_ticks=5,
    mm_min_order_size=3,
    mm_skew_beta=0,
    mm_price_skew=4,
    mm_level_spacing=5,
    mm_spread_alpha=0.75,
    mm_backstop_quantity=0,
    mm_cancel_limit_delay=50,  # 50 nanoseconds
    ##value agent
    val_kappa=1.67e-15,
    val_vol=1e-8,
    val_lambda_a=1e-12,
    computation_delay=0,
):
    symbol = ticker

    # Setting numpy seed
    np.random.seed(seed)

    ########################################################################################################################
    ############################################### AGENTS CONFIG ##########################################################    
    period_start = str(np.random.choice(period_starts))
    print("Period of start:", period_start)  
    # Convert to nanoseconds
    start_time = datetime_str_to_ns(period_start)

    mkt_open = start_time
    mkt_close = start_time + str_to_ns("01:00:00")
    noise_mkt_open = mkt_open - str_to_ns("00:00:30")
    agent_count, agents, agent_types = 0, [], []
    
    # num_noise_agents = int((mkt_close - mkt_open) / str_to_ns("01:00:00") * num_noise_agents)

    # Oracle
    symbols = {
        symbol : {
            'fundamental_file_path': fundamental_file_path,
            'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2**32, dtype='uint64'))
        }
    }
    oracle = ExternalFileOracle(symbols)

    # Params for value agents.
    r_bar = oracle.fundamentals[symbol].values[0]
    sigma_n = r_bar / 3000
    
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
                computation_delay=computation_delay,
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
                    #order_size_model=ORDER_SIZE_MODEL,
                    random_state=np.random.RandomState(
                        seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
            )
            for j in range(agent_count, agent_count + num_noise_agents)
        ]
    )
    agent_count += num_noise_agents
    agent_types.extend(['NoiseAgent'])

    # 3) Value Agents
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
                kappa=val_kappa,
                lambda_a=val_lambda_a,
                log_orders=log_orders,
                order_size_model=ORDER_SIZE_MODEL,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
            )
            for j in range(agent_count, agent_count + num_value_agents)
        ]
    )
    agent_count += num_value_agents
    agent_types.extend(['ValueAgent'])

    # 4) Adaptive Market Maker
    mm_params = num_mm * [
        (mm_window_size, mm_pov, mm_num_ticks, mm_wake_up_freq, mm_min_order_size)
    ]

    num_mm_agents = len(mm_params)
    mm_cancel_limit_delay = 50  # 50 nanoseconds

    agents.extend(
        [
            AdaptiveMarketMakerAgent(
                id=j,
                name="ADAPTIVE_POV_MARKET_MAKER_AGENT_{}".format(j),
                type="AdaptivePOVMarketMakerAgent",
                symbol=symbol,
                starting_cash=starting_cash,
                pov=mm_params[idx][1],
                min_order_size=mm_params[idx][4],
                window_size=mm_params[idx][0],
                num_ticks=mm_params[idx][2],
                wake_up_freq=str_to_ns(mm_params[idx][3]),
                cancel_limit_delay=mm_cancel_limit_delay,
                skew_beta=mm_skew_beta,
                level_spacing=mm_level_spacing,
                spread_alpha=mm_spread_alpha,
                backstop_quantity=mm_backstop_quantity,
                log_orders=log_orders,
            )
            for idx, j in enumerate(range(agent_count, agent_count + num_mm_agents))
        ]
    )
    agent_count += num_mm_agents
    agent_types.extend("POVMarketMakerAgent")

    # 5) Momentum Agents
    agents.extend(
        [
            MomentumAgent(
                id=j,
                name="MOMENTUM_AGENT_{}".format(j),
                symbol=symbol,
                starting_cash=starting_cash,
                min_size=1,
                max_size=4,
                wake_up_freq=str_to_ns(m_wake_up_freq),
                log_orders=log_orders,
                #order_size_model=ORDER_SIZE_MODEL,
            )
            for j in range(agent_count, agent_count + num_momentum_agents)
        ]
    )
    agent_count += num_momentum_agents
    agent_types.extend("MomentumAgent")


    ########################################################################################################################
    ########################################### KERNEL AND OTHER CONFIG ####################################################

    # extract kernel seed here to reproduce the state of random generator in old version
    random_state_kernel = np.random.RandomState(
        seed=np.random.randint(low=0, high=2**32, dtype="uint64")
    )

    kernelStartTime = int(pd.to_datetime(period_start.split(" ")[0]).to_datetime64())
    kernelStopTime = mkt_close + str_to_ns('1s')

    latency_model = generate_latency_model(agent_count)
    default_computation_delay = 50  # 50 nanoseconds

    return {
        "ticker": symbol,
        "mkt_close": mkt_close,
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

    
