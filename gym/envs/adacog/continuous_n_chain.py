import math
import random
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class ContinuousNChainEnv(gym.Env):
    """n-Chain environment
    This is a continuous version of the n-chain environment:
    The end of the chain, however, presents a large reward, and by moving
    'forward' at the end of the chain this large reward can be repeated.
    At each action, there is a small probability that the agent 'slips' and the
    opposite transition is instead taken.
    The observed state is the current state in the chain (0 to n-1).
    This environment is described in section 6.1 of:
    A Bayesian Framework for Reinforcement Learning by Malcolm Strens (2000)
    http://ceit.aut.ac.ir/~shiry/lecture/machine-learning/papers/BRL-2000.pdf
    """
    def __init__(self, n=5, alpha=10, beta=10, action_power = 6, max_step = 25):
        self.n = n
        self.state = 0  # Start at beginning of the chain

        self.action_power = 6 # Higher values make moving forward less likely, more difficult task. 

        self.min_action = -1.0
        self.max_action = 1.0
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(1,), dtype=np.float32)

        self.alpha = alpha
        self.beta = beta
        self.hiddenValues = np.random.beta(a = self.alpha, b = self.beta, size = self.n)
        
        self.max_step = max_step
        self.timestep = 0
        
        self.observation_space = spaces.Discrete(self.n)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def updatDistribution(self, alpha = None, beta = None):
        if(alpha is not None):
            self.alpha = alpha

        if(beta is not None):
            self.beta = beta
        
        self.resample()

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
            self.state -= 1
            # No Slipping
            #self.state -= 0
        else:
            self.state = 0
        
        if(self.state >= self.n):
            done = True
            reward = 0
            print("done")
        
        if(self.timestep >= self.max_step):
            done = True
            reward = -1

        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        self.resample()
        self.timestep = 0

        return np.array([self.state])