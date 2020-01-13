import math
import random
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class ContinuousNChainEnv(gym.Env):
    """n-Chain environment
    This is a continuous version of the n-chain environment:
    A Bayesian Framework for Reinforcement Learning by Malcolm Strens (2000)
    http://ceit.aut.ac.ir/~shiry/lecture/machine-learning/papers/BRL-2000.pdf
    """
    def __init__(self, n=5, alpha=10, beta=25, action_power = 12, max_step = 100): 
        self.n = n
        self.state = 0  # Start at beginning of the chain

        self.action_power = action_power # bigger values make moving forward less likely, more difficult task. 

        self.min_action = -1.0
        self.max_action = 1.0
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(1,), dtype=np.float32)

        self.alpha = alpha
        self.beta = beta
        self.base_alpha = alpha
        self.base_beta = beta
        self.hiddenValues = np.random.beta(a = self.alpha, b = self.beta, size = self.n)
        
        self.max_step = max_step
        self.timestep = 0
        
        self.observation_space = spaces.Discrete(self.n)
        self.seed()

    def randomize(self):
        base_alpha = self.base_alpha
        base_beta = self.base_beta

        alpha = random.uniform(base_alpha * 0.9 , base_alpha * 1.1)
        beta = random.uniform(base_beta * 0.9 , base_beta * 1.1)

        #print("alpha ", alpha, " beta ", beta)

        self.updatDistribution(alpha = alpha, beta = beta)
        self.reset()
    
    def randomize_extreme(self):
        base_alpha = self.base_alpha
        base_beta = self.base_beta

        if(np.random.rand(0) > 0.5):
            alpha = random.uniform(base_alpha * .75 , base_alpha * 0.9 )
            beta = random.uniform(base_beta * 1.1 , base_beta * 1.25 )
        else:
            alpha = random.uniform(base_alpha * 1.1 , base_alpha * 1.25 )
            beta = random.uniform(base_beta * .75 , base_beta * 0.9 )
        
        #print("alpha ", alpha, " beta ", beta)

        self.updatDistribution(alpha = alpha, beta = beta)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def updatDistribution(self, alpha = None, beta = None):
        if(alpha is not None):
            self.alpha = alpha

        if(beta is not None):
            self.beta = beta
        
        self.resample()

    def setHiddenValues(self, hiddenValues):
        self.hiddenValues = hiddenValues
    
    def getHiddenValues(self):
        return self.hiddenValues

    def resample(self):
        self.hiddenValues = np.random.beta(a = self.alpha, b = self.beta, size = self.n)

    def step(self, action):
        action = (action + 1)/(2) # Normalize the [-1,1] action to [0,1], gym required actions spaces to be symmetric. 
        
        hiddenValue = self.hiddenValues[self.state] # Get the current state's hidden value 
        p_forward = (1 - np.abs((action - hiddenValue))) ** self.action_power  # Probability of moving forward
        reward = -1
        done = False

        self.timestep += 1 
        
        if(random.random() < p_forward):
            self.state += 1
        elif(self.state > 0):
            # Slipping
            # self.state -= 1
            # No Slipping
            self.state -= 0
        else:
            self.state = 0
        
        if(self.state >= self.n):
            done = True
            reward = 0
            #print("done")
        
        if(self.timestep >= self.max_step):
            done = True
            reward = -1

        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        # self.resample() # don't resample
        self.timestep = 0

        return np.array([self.state])