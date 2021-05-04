import gym
import air_rev
import numpy as np


from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from policy_evaluate import evaluate

modelAvgs = []
numEnvs = 10
for i in range(numEnvs):
    #env = make_vec_env('air_rev-v0', n_envs=1)
    env = gym.make('air_rev-v0')

    model = PPO(MlpPolicy, env, verbose=1, n_steps=2048, gamma = 1)

    model.learn(total_timesteps=100000)

    model.save("ar_1")

    #env = gym.make('air_rev-v0')
    env.reset()
    n_episodes = 1000
    res_mean, res_std = evaluate(model, env, n_episodes)
    print(res_mean,'+/-',1.96*res_std/np.sqrt(n_episodes))

    modelAvgs += [res_mean]
print(np.mean(modelAvgs), '+/-', 1.96*np.std(modelAvgs)/np.sqrt(numEnvs))

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

