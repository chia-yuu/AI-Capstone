# black jack, q learning
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt

total_reward = []
total_q_val = []

# Agent to perform Q learning
class Agent():
    def __init__(self, env, epsilon=0.05, lr=0.01, gamma=0.95):
        self.env = env
        self.epsilon = epsilon  # explore or exploit
        self.lr = lr
        self.gamma = gamma

        self.q_table = np.zeros((32, 11, 2, env.action_space.n))
    
    # choose an action with the given state
    def choose_action(self, state, step):
        action = None
        tmp = np.random.uniform(0, 1)
        # epsilon = 0.01 + 0.99 * np.exp(-0.01 * step)
        if tmp < self.epsilon:
            # random action
            action = env.action_space.sample()
        else:
            # action with max Q
            action = np.argmax(self.q_table[state[0], state[1], state[2], :])
        return action
    
    # update Q table based on reward & state after taking action
    def update_table(self, state, action, reward, next_state, done):
        qsa = self.q_table[state[0], state[1], state[2], action]
        mx = np.max(self.q_table[next_state[0], next_state[1], next_state[2], :])
        self.q_table[state[0], state[1], state[2], action] = qsa + self.lr * (reward + self.gamma * mx - qsa)
    
    def epsilon_decay(self):
        self.epsilon = max(0.1, self.epsilon - (1.0 / 50000))


def train(env):
    agent = Agent(env)
    episode = 100000
    rewards = []
    q_val = []
    reward_record = defaultdict(int)
    action_record = [0, 0]
    for ep in tqdm(range(episode)):
        state, info = env.reset()
        r_cnt = 0
        cnt = 0
        
        while True:
            cnt += 1
            action = agent.choose_action(state, cnt)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update_table(state, action, reward, next_state, terminated or truncated)
            r_cnt += reward
            state = next_state
            reward_record[reward] += 1
            action_record[action] += 1

            if terminated or truncated:
                rewards.append(r_cnt)
                agent.epsilon_decay()
                break

        if (ep + 1) % 5 == 0:
            q_val.append(agent.q_table.mean())

    np.save("./Table/task3-1.npy", agent.q_table)
    total_reward.append(rewards)
    total_q_val.append(q_val)
    print(f"actions: {action_record}, ", end=' ')
    print(f"Average reward = {np.mean(rewards)}")


def test(env):
    agent = Agent(env)
    agent.q_table = np.load("./Table/task3-1.npy")
    rewards = []
    reward_record = defaultdict(int)
    action_record = [0, 0]

    for _ in tqdm(range(100)):
        state, info = agent.env.reset()
        cnt = 0
        while True:
            if args.visualize:
                agent.env.render()
            action = np.argmax(agent.q_table[state[0], state[1], state[2], :])
            next_state, reward, terminated, truncated, info = agent.env.step(action)
            cnt += reward
            state = next_state
            reward_record[reward] += 1
            action_record[action] += 1

            if terminated or truncated:
                rewards.append(cnt)
                break
    print(f"actions: {action_record}, ", end=' ')
    print(f"Average reward = {np.mean(rewards)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testOnly", action='store_true')
    parser.add_argument("--visualize", action='store_true')
    args = parser.parse_args()

    env = gym.make('Blackjack-v1', natural=False, sab=False)
    if args.visualize:
        env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode='human')  # visualize
    os.makedirs("./Table", exist_ok=True)
    os.makedirs("./Rewards", exist_ok=True)
    os.makedirs("./Q Values", exist_ok=True)

    if not args.testOnly:
        for i in range(5):
            print(f"## {i+1} training progress")
            train(env)
            np.save("./Rewards/task3-1.npy", np.array(total_reward))
            np.save("./Q Values/task3-1.npy", np.array(total_q_val))
    
    print("## testing progress")
    test(env)

    env.close()
