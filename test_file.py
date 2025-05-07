import gym
import abides_gym
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from abides_core.utils import plot_state_hist, save_pickle
import time

env = gym.make(
        "markets-execution-v0",
        background_config="FX_test_execution",
    )

env.seed(0)

PnL = []
for j in range(1):
    start = time.time()
    num_features = 12+1
    num_actions = 7
    num_steps = 1000
    data_history = []
    reward_history = []
    current_times = []
    
    initial_state = env.reset()
    done = False
    i = 0

    while not done and i < num_steps:
        action = np.random.randint(0,7)
        state, reward, done, info = env.step(action)
        data_history.append(list(state.flatten()) + [info["fundamental_value"]])
        current_times.append(info["current_time"])
        reward_history.append([reward, env.pnl, env.direc])


        if i % 500 == 0:
            print(i)
            arr = np.array(reward_history)
            plt.figure(figsize=(8,4), dpi=200)
            plt.plot(current_times, arr[:,1], label="PnL")
            plt.plot(current_times, arr[:,2], label="Directional")
            plt.plot(current_times, arr[:,0], label="Total")
            plt.legend()
            plt.savefig("results/reward.png", dpi=200)
            plt.close()

            dh = np.array(data_history)

            #plot_state_hist(dh, time_index=current_times, sim_date=pd.Timestamp(env.kernelStartTime))
    
        i += 1
    print(np.sum(arr[:,1]), np.sum(arr[:,2]))
    print(time.time() - start)

#print(pd.Series(dh[:,3]).diff(1).count(0))
# arr = np.array(PnL)
# print(round(arr[arr == 0].shape[0]/arr.shape[0], 2))
# plt.figure(figsize=(8,4), dpi=200)
# plt.hist(PnL, bins=100)
# plt.title("PnL Reward")
# plt.xlabel("Reward")
# plt.ylabel("Frequency")
# plt.savefig("results/scaling_PnL/PnL_hist_085.png", dpi=200)
