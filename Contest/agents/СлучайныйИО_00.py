# agents/ЖадныйИО_группа_agent.py
import random
import numpy as np


def agent(observation, configuration):
    """Случайный выбор бандита."""
    return random.randrange(configuration['num_bandits'])