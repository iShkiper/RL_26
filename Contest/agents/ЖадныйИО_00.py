# agents/ЖадныйИО_группа_agent.py
import random
import numpy as np


def agent(observation, configuration):
    """
    Код агента
    """
    """Жадный: всегда выбирает бандит с наибольшей средней наградой (по своим наблюдениям)."""
    n = observation['n_pulls']
    s = observation['sum_rewards']
    if np.min(n) == 0:
        return int(np.argmin(n))
    means = s / n
    return int(np.argmax(means))
