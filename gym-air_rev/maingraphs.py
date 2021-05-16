import gym
import air_rev
import numpy as np
import matplotlib.pyplot as plt

import torch as th
from ppo import PPO
from stable_baselines3.ppo import MlpPolicy
# from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from policy_evaluate import evaluate

modelAvgs = []
modelStds = []
numEnvs = 1
for i in range(numEnvs):
    print("round number " + str(i))
    #env = make_vec_env('air_rev-v0', n_envs=1)
    env = gym.make('air_rev-v0')

    l = len(env.revenue)
    print(sum(env.revenue[l//2:]))
    # policy_kwargs = dict(optimizer_class=th.optim.SGD)
    # model = PPO(MlpPolicy, env, policy_kwargs = policy_kwargs, learning_rate=0.1, verbose=0, n_steps=1000, gamma = 1.0)
    model = PPO(MlpPolicy, env, learning_rate=0.003, verbose=0, n_steps=1000, gamma = 1.0)

    # loop 10000 times: do model.learn for one timestep, do 10 episodes
    # of evaluate, save mean and std, plot the graph of the mean and std
    # over time. Finish off with a 100 episode evaluate of the final model?
    timeSteps=500
    plotMeans = []
    plotStds = []
    plotTimes = range(timeSteps)
    for j in range(timeSteps):
        if j % 10 == 0:
            print("round " + str(j))
        model.learn(total_timesteps=100)

        #model.save("ar_1")

        #env = gym.make('air_rev-v0')
        env.reset()
        n_episodes = 30
        res_mean, res_std = evaluate(model, env, n_episodes)
        plotMeans += [res_mean]
        # print(res_mean)
        plotStds += [res_std/np.sqrt(n_episodes)]

    plt.plot(plotTimes, plotMeans)
    plt.show()
    plt.plot(plotTimes, plotStds)
    plt.show()

    print(res_mean,'+/-',1.96*res_std/np.sqrt(n_episodes))

    modelAvgs += [res_mean]
    modelStds += [res_std]
print(max(modelAvgs), min(modelAvgs), np.mean(modelStds)/np.sqrt(n_episodes))

#env = CrissCross(load = 0.5)#gym.make('criss_cross-v0')
# n_steps = 20000
# av_reward = np.zeros(n_steps)
# av_r = 0.
# env.reset()
# for i in range(n_steps):
#     # Random action
#     k = 0
#     done = False
#     env.reset()
#     av_r = 0.
#     while not done:
#         action = env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#         av_r = reward +  av_r
#         k = k + 1
#
#     av_reward[i]=av_r
#
#
# print(np.mean(av_reward),'+-',1.96*np.std(av_reward)/np.sqrt(n_steps))

