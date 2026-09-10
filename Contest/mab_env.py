import numpy as np
import random
import math
from collections import defaultdict

# ------------------------------------------------------------
# 1. Среда Dynamic Multi-Armed Bandit
# ------------------------------------------------------------
class DynamicMABEnv:
    def __init__(self, bandit_probs=None, num_bandits=100, num_rounds=2000,
                 seed=None, boost_prob=0.0, boost_amount=0.0):
        self.num_bandits = num_bandits
        self.num_rounds = num_rounds
        self.seed = seed
        self.boost_prob = boost_prob
        self.boost_amount = boost_amount
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if bandit_probs is None:
            self.bandit_probs = np.random.beta(1, 1, size=num_bandits)
        else:
            self.bandit_probs = np.array(bandit_probs)
            assert len(self.bandit_probs) == num_bandits

        self.n_pulls = [np.zeros(num_bandits, dtype=int) for _ in range(2)]
        self.sum_rewards = [np.zeros(num_bandits, dtype=int) for _ in range(2)]
        self.actions_history = [[], []]
        self.current_step = 0

    def reset(self):
        self.n_pulls = [np.zeros(self.num_bandits, dtype=int) for _ in range(2)]
        self.sum_rewards = [np.zeros(self.num_bandits, dtype=int) for _ in range(2)]
        self.actions_history = [[], []]
        self.current_step = 0
        return self._get_observations()

    def _get_observations(self):
        obs = []
        for agent_idx in range(2):
            other_idx = 1 - agent_idx
            obs.append({
                'n_pulls': self.n_pulls[agent_idx].copy(),
                'sum_rewards': self.sum_rewards[agent_idx].copy(),
                'step': self.current_step,
                'my_actions': self.actions_history[agent_idx].copy(),
                'other_actions': self.actions_history[other_idx].copy(),
            })
        return obs

    def step(self, actions):
        assert len(actions) == 2, "Ожидается два действия"
        rewards = []
        for agent_idx, action in enumerate(actions):
            self.actions_history[agent_idx].append(action)

        for agent_idx, action in enumerate(actions):
            if not (0 <= action < self.num_bandits):
                raise ValueError(f"Агент {agent_idx} выбрал недопустимый бандит {action}")

            prob = self.bandit_probs[action]
            reward = 1 if np.random.random() < prob else 0
            self.bandit_probs[action] *= 0.97

            self.n_pulls[agent_idx][action] += 1
            self.sum_rewards[agent_idx][action] += reward
            rewards.append(reward)

        # ---- БУСТ ----
        if self.boost_prob > 0 and np.random.random() < self.boost_prob:
            bandit = np.random.randint(self.num_bandits)
            self.bandit_probs[bandit] = min(1.0, self.bandit_probs[bandit] + self.boost_amount)

        self.current_step += 1
        done = (self.current_step >= self.num_rounds)
        observations = self._get_observations()
        info = {'bandit_probs': self.bandit_probs}
        return observations, rewards, done, info


# ------------------------------------------------------------
# 3. Запуск одного эпизода
# ------------------------------------------------------------
def run_episode(agents, env, episode_idx=0, verbose=False):
    """
    Запускает один эпизод.
    agents: список из двух функций агентов в исходном порядке (агент0, агент1).
    episode_idx: номер эпизода (используется для детерминированной смены порядка).
    Возвращает суммарные награды для агентов в исходном порядке.
    """
    # Определяем порядок хода: если номер эпизода нечётный, меняем местами
    if episode_idx % 2 == 1:
        order = [1, 0]   # сначала агент1, потом агент0
        swapped = True
    else:
        order = [0, 1]   # стандартный порядок
        swapped = False

    # Создаём список агентов в порядке хода
    agents_in_order = [agents[order[0]], agents[order[1]]]

    # Формируем конфигурации для каждого агента (один раз для всего эпизода)
    configs = [
        {'num_bandits': env.num_bandits, 'num_rounds': env.num_rounds, 'position': 0},
        {'num_bandits': env.num_bandits, 'num_rounds': env.num_rounds, 'position': 1}
    ]

    observations = env.reset()
    total_rewards = [0, 0]   # накапливаем в исходном порядке
    done = False
    while not done:
        actions = []
        for pos, agent in enumerate(agents_in_order):
            obs = observations[pos].copy()
            action = agent(obs, configs[pos])   # используем заранее подготовленный config
            actions.append(action)

        observations, rewards, done, info = env.step(actions)
        # rewards соответствуют порядку agents_in_order
        if not swapped:
            total_rewards[0] += rewards[0]
            total_rewards[1] += rewards[1]
        else:
            total_rewards[0] += rewards[1]   # rewards[0] — это награда агента1
            total_rewards[1] += rewards[0]

        if verbose:
            print(f"Round {env.current_step}: actions {actions}, rewards {rewards}, total {total_rewards}")

    return total_rewards
    

from tqdm import tqdm

# ------------------------------------------------------------
# 4. Турнир: сравнение нескольких пар агентов
# ------------------------------------------------------------
def evaluate_agents(agent_pairs, num_bandits=100, num_rounds=2000, num_episodes=50, seed=None, 
                    boost_prob=0.0, boost_amount=0.0):
    """
    Оценивает список пар агентов на num_episodes случайных наборах вероятностей.
    agent_pairs: список кортежей (name1, agent_func1, name2, agent_func2)
    Возвращает словарь с результатами.
    """
    results = {}
    for pair_idx, (name1, agent1, name2, agent2) in enumerate(agent_pairs):
        rewards1 = []
        rewards2 = []
        for ep in tqdm(range(num_episodes)):
            # Генерируем новые начальные вероятности для каждого эпизода
            env = DynamicMABEnv(num_bandits=num_bandits, num_rounds=num_rounds,
                                seed=None if seed is None else seed + ep + pair_idx*1000,
                               boost_prob=boost_prob, boost_amount=boost_prob)
            total1, total2 = run_episode([agent1, agent2], env, verbose=False)
            rewards1.append(total1)
            rewards2.append(total2)

        mean1 = np.mean(rewards1)
        mean2 = np.mean(rewards2)
        results[f"{name1} vs {name2}"] = {
            name1: mean1,
            name2: mean2,
            'diff': mean1 - mean2
        }
    return results