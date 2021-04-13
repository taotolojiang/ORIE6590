import gym
from ar_env import AirlineRevenue
import numpy as np
from policy_evaluate import evaluate

avg_reward, mean_avg_reward,std_95 = evaluate([])
print(mean_avg_reward,'+-',std_95)
