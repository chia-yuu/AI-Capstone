# space invader
import gymnasium as gym
import ale_py
import numpy as np
from tqdm import tqdm
import os

total_reward = []


# Agent to perform Q learning
class Agent():
    def __init__(self, env, epsilon=0.05, lr=0.8, gamma=0.9):
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
        self.q_table = self.q_table + self.lr * (reward + self.gamma * np.max(self.q_table[next_state[0], next_state[1], next_state[2], :]) - self.q_table[state[0], state[1], state[2], action])
        # save q table if done
        if done:
            np.save("./Table/space_invader.npy", self.q_table)


def train(env):
    agent = Agent(env)
    episode = 3000
    rewards = []
    for ep in tqdm(range(episode)):
        state, info = env.reset()
        cnt = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            print(next_state)
            raise NotImplementedError
            agent.update_table(state, action, reward, next_state, terminated or truncated)
            cnt += reward
            state = next_state

            if terminated or truncated:
                rewards.append(cnt)
                break

    total_reward.append(rewards)


def test(env):
    agent = Agent(env)
    agent.q_table = np.load("./Table/space_invader.npy")
    rewards = []

    for _ in range(100):
        state, info = agent.env.reset()
        cnt = 0
        while True:
            action = np.argmax(agent.q_table[state[0], state[1], state[2]])
            next_state, reward, terminated, truncated, info = env.step(action)
            cnt += reward
            state = next_state

            if terminated or truncated:
                rewards.append(cnt)
                break
    print(f"Average reward = {np.mean(rewards)}")


if __name__ == "__main__":
    gym.register_envs(ale_py)
    env = gym.make("ALE/SpaceInvaders-v5")
    os.makedirs("./Table", exist_ok=True)

    for i in range(5):
        print(f"## {i+1} training progress")
        train(env)
    
    test(env)

    os.makedirs("./Rewards", exist_ok=True)

    np.save("./Rewards/space_invader.npy", np.array(total_reward))

    env.close()
