import gym
import numpy as np
import sys
import copy
import math

class AirRev(gym.Env):

    def __init__(self, l =3, epoch = 20, capVal = 2.0):
        # hi-lo demand hyperparams
        # self.a = 5
        # self.rho = 1

        # checks if we should end
        self.done = False

        self.m = 2 * l
        self.n = 2 * (l**2 + l)  # slide 11 from lecture 5 has a typo (should be + instead of -)
        num_iter = int(self.n/2)

        self.timeSteps = 0
        self.capa = [capVal]*self.m
        self.capacities = np.array(self.capa)  
        self.demands = np.identity(self.m)
        for i in range(l):
            for j in range(l):
                if i != j:
                    demand_col = np.zeros((self.m, 1))
                    demand_col[2 * i + 1] = 1.0
                    demand_col[2 * j] = 1.0
                    self.demands = np.append(self.demands, demand_col, axis = 1)
        self.demands = np.append(self.demands, self.demands, axis = 1)
        # lowFares = np.random.randint(15,50,self.n//2)
        # self.revenue = np.append(lowFares, 5*lowFares)
        # print(self.revenue)
        #self.revenue = np.asarray([[1.0,2.0][j] for j in range(2) for i in range(num_iter)])
        self.epoch = epoch  # the number of time steps we have to finish within
        # itineraryDemands = np.random.uniform(0,1,self.n//2)
        # scaleTerm = sum(itineraryDemands)
        # self.itinDemds = 0.8*itineraryDemands/scaleTerm
        # demdsWoNoArrival = np.append(0.75*self.itinDemds, 0.25*self.itinDemds)
        # self.probabilities = np.append(demdsWoNoArrival, np.asarray(0.2))
        # print(self.probabilities)
        
        # for 3, 20, 2
        self.revenue = np.array([33, 28, 36, 34, 17, 20, 39, 24, 31, 19, \
                                 30, 48, 165, 140, 180, 170, 85, 100,    \
                                 195, 120, 155, 95, 150, 240])
        self.probabilities = np.array([0.01327884, 0.02244177, 0.07923761, \
                                       0.0297121,  0.02654582, 0.08408091, \
                                       0.09591975, 0.00671065, 0.08147508, \
                                       0.00977341, 0.02966204, 0.121162,   \
                                       0.00442628, 0.00748059, 0.02641254, \
                                       0.00990403, 0.00884861, 0.02802697, \
                                       0.03197325, 0.00223688, 0.02715836, \
                                       0.0032578,  0.00988735, 0.04038733, \
                                       0.2])

        # the final entry accounts for the probability of no arrival


        self.state = np.array(self.capa)  # Start at beginning of the chain
        self.action_space = gym.spaces.MultiBinary(self.n)
        self.observation_space = gym.spaces.MultiDiscrete([capVal + 1]*self.m) #capa + 1 since strictly less than
        self.seed()
        metadata = {'render.modes': ['ansi']}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if not self.done:
            self.timeSteps += 1

            # change probabilities
            # log base a
            # hiFareDelta = math.log(self.a+(self.epoch - self.timeSteps)*self.rho, self.a)
            # lowFareDelta = math.log(self.a+(self.timeSteps - 1)*self.rho, self.a)
            # demdsWoNoArrival = np.append(lowFareDelta*self.itinDemds, hiFareDelta*self.itinDemds)
            # self.probabilities = np.append(demdsWoNoArrival, np.asarray(0.2))

            assert self.action_space.contains(action)

            reward = 0.0

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
                        if self.state[i] - dems[i] < 0:
                            bookable = False
                    if bookable:
                        #print("-----")
                        #print(self.state)
                        #print(dems)
                        #print("-----")

                        self.state -= dems
                        reward = self.revenue[activity]
                        #reward = np.exp(-self.timeSteps/np.sum(self.state))*revThisRound
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
            self.done = ((np.sum(self.state) == 0) or (self.timeSteps == self.epoch))
            
        return self.state, reward, self.done, {}

    def render(self, mode='ansi'):
        outfile = sys.stdout if mode == 'ansi' else super(AirRev, self).render(mode=mode)
        outfile.write(np.array2string(self.state))


    def reset(self):
        self.state = np.array(self.capa)
        self.done = False
        self.timeSteps = 0
        return self.state
