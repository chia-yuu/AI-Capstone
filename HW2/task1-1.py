# space invader, q learning
import gymnasium as gym
import ale_py
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict
import argparse

total_reward = []

# Agent to perform Q learning
class Agent():
    def __init__(self, env, epsilon=0.05, lr=0.5, gamma=0.9):
        self.env = env
        self.epsilon = epsilon  # explore or exploit
        self.lr = lr
        self.gamma = gamma

        self.q_table = np.zeros((210, 160, 3, env.action_space.n))
    
    # choose an action with the given state
    def choose_action(self, state):
        action = None
        tmp = np.random.uniform(0, 1)
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

        # save q table if done
        if done:
            np.save("./Table/task1-1.npy", self.q_table)


def train(env):
    agent = Agent(env)
    episode = 300
    rewards = []
    reward_record = defaultdict(int)
    action_record = [0, 0, 0, 0, 0, 0]
    for ep in tqdm(range(episode)):
        state, info = env.reset()
        cnt = 0
        last_shoot = 0  # reward got from the last shoot
        pre_action = -1 # previous action when shoot (1, 4, 5)

        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update_table(state, action, reward, next_state, terminated or truncated)
            cnt += reward
            state = next_state
            reward_record[reward] += 1
            action_record[action] += 1

            if terminated or truncated:
                rewards.append(cnt)
                break

        if (ep + 1) % 100 == 0:
            agent.lr -= 0.05

    total_reward.append(rewards)
    print(f"actions: {action_record}, ", end=' ')
    # print(f"rewards: {reward_record}")
    print(f"Average reward = {np.mean(rewards)}")


def test(env):
    agent = Agent(env)
    agent.q_table = np.load("./Table/task1-1.npy")
    rewards = []
    reward_record = defaultdict(int)
    action_record = [0, 0, 0, 0, 0, 0]

    for _ in range(100):
        state, info = agent.env.reset()
        cnt = 0
        while True:
            # agent.env.render()
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
    args = parser.parse_args()

    gym.register_envs(ale_py)
    # env = gym.make("ALE/SpaceInvaders-v5", render_mode='human')  # visualize
    env = gym.make("ALE/SpaceInvaders-v5")
    os.makedirs("./Table", exist_ok=True)

    if not args.testOnly:
        for i in range(5):
            print(f"## {i+1} training progress")
            train(env)
        os.makedirs("./Rewards", exist_ok=True)
        np.save("./Rewards/task1-1.npy", np.array(total_reward))
    
    print("## testing progress")
    test(env)

    env.close()
