import gym
import air_rev
import numpy as np
from stable_baselines3 import PPO

def evaluate(model, env, n_steps=500):
    avg_reward = np.zeros(n_steps)
    avg_timesteps = np.zeros(n_steps)
    for i in range(n_steps):
        k = 0
        done = False
        obs = env.reset()
        avg_r = 0.
        env.seed()
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            avg_r = reward + avg_r
            k = k + 1
        avg_reward[i]=avg_r
        avg_timesteps[i] = env.timeSteps
    # print(avg_reward)
    # print(max(avg_reward))
    # print(min(avg_reward))
    # print(len(list(filter(lambda x: x < 20, avg_timesteps))))

    return np.mean(avg_reward), np.std(avg_reward)