import math
import random
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class HanabiEnv(gym.Env):
    """Hanabi Card Game Environment
    https://www.spillehulen.dk/media/102616/hanabi-card-game-rules.pdf
    """
    def __init__(self, players=4, colors=4): 

        # Creates a card board to hold cards plaeyd by players
        Board = [[0 for x in range(5)] for y in range(colors)] 
        Hands = [[0 for x in range(5)] for y in range(colors)] 