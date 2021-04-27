import gym
import numpy as np
import sys
import copy

class AirRev(gym.Env):

    def __init__(self, load=0.5, l =4):
        self.m = 2 * l
        self.n = 2 * (l**2 + l)  # slide 11 from lecture 5 has a typo (should be + instead of -)
        num_iter = int(self.n/2)

        capa = [1.0]*self.m
        self.capacities = np.array(capa)  
        self.demands = np.identity(self.m)
        for i in range(l):
            for j in range(l):
                if i != j:
                    demand_col = np.zeros((self.m, 1))
                    demand_col[2 * i + 1] = 1.0
                    demand_col[2 * j] = 1.0
                    self.demands = np.append(self.demands, demand_col, axis = 1)
        self.demands = np.append(self.demands, self.demands, axis = 1)
        self.revenue = np.asarray([1.0]*self.m + [2.0]*(self.n-self.m)+[2.0]*self.m + [4.0]*(self.n-self.m))
        #self.revenue = np.asarray([[1.0,2.0][j] for j in range(2) for i in range(num_iter)])
        self.epoch = 50  # the number of time steps we have to finish within
        self.probabilities = np.asarray([[3/4*1.8/self.n,1/4*1.8/self.n][j] for j in range(2) for i in range(num_iter)] + [0.1])
        # the final entry accounts for the probability of no arrival


        self.state = np.array(capa)  # Start at beginning of the chain
        self.action_space = gym.spaces.MultiBinary(self.n)
        self.observation_space = gym.spaces.MultiDiscrete([2.0]*self.m) #capa + 1 since strictly less than
        self.seed()
        metadata = {'render.modes': ['ansi']}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        self.reward = 0.0

        activity = np.random.choice(range(self.n+1), 1, list(self.probabilities))[0]
        # if we don't have the action correlated with "no customer arriving"
        if activity != self.n:

            if action[activity] == 1:
                # activity is what class of customer did arrive
                # demands for this class is the activity'th column
                dems = self.demands[:,activity]
                # check if we can book this customer
                bookable = True
                # check all demands
                for i in range(len(dems)):
                    # check there is enough seats on flight to meet demand
                    if self.state[i] < dems[i]:
                        bookable = False
                if bookable:
                    #print("-----")
                    #print(self.state)
                    #print(dems)
                    #print("-----")

                    self.state -= dems
                    self.reward = self.revenue[activity]
                    #print(self.reward)
                else:
                    pass
                    #print("demand exceed capacity")
            elif action[activity] == 0:
                # Do nothing
                pass
            else:
                # error
                print("action space not binary")
        else:
            pass
            #print("No customer arrives")

        done = (np.sum(self.state) == 0) # with more complicated demand allocations, we might need additional cases
        return self.state, self.reward, done, {}

    def render(self, mode='ansi'):
        outfile = sys.stdout if mode == 'ansi' else super(AirRev, self).render(mode=mode)
        outfile.write(np.array2string(self.state))



    def reset(self):
        self.state = np.array([1.0]*self.m)
        return self.state
