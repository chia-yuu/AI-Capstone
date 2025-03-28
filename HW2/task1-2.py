# space invader, DQN
import gymnasium as gym
import ale_py
import numpy as np
import torch.optim.adam
from tqdm import tqdm
import os
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

total_reward = []

class Replay_buffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def insert(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.memory)
    
# The Q network
class Model(nn.Module):
    def __init__(self, n_action, hidden_layer_size=50):
        super(Model, self).__init__()
        self.input_state = 210 * 160 * 3    # observation space
        self.n_action = n_action            # action space
        self.fc1 = nn.Linear(self.input_state, 32)
        self.fc2 = nn.Linear(32, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, n_action)
    
    def forward(self, state):
        state = state.view(state.size(0), -1)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


# Agent to perform DQN
class Agent():
    def __init__(self, env, epsilon=0.05, lr=0.8, gamma=0.9, batch_size=32, capacity=10000):
        self.env = env
        self.epsilon = epsilon  # explore or exploit
        self.lr = lr
        self.gamma = gamma

        self.n_action = 6
        self.count = 0      # when to update net
        self.batch_size = batch_size
        self.capacity = capacity
        
        self.buffer = Replay_buffer(self.capacity)
        self.eval_net = Model(self.n_action)
        self.target_net = Model(self.n_action)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
    
    # choose an action with the given state
    def choose_action(self, state):
        with torch.no_grad():
            action = None
            tmp = np.random.uniform(0, 1)
            if tmp < self.epsilon:
                # random action
                action = env.action_space.sample()
            else:
                # action with max Q
                s = torch.tensor(state, dtype=torch.float)
                q = self.eval_net(s)
                action = torch.argmax(q).item()
        return action
    
    # update Q table based on reward & state after taking action
    def update_table(self):
        if self.count%100 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        t_state, action, t_reward, t_next_state, done = self.buffer.sample(self.batch_size)
        state = torch.tensor(t_state, dtype=torch.float)
        reward = torch.tensor(t_reward, dtype=torch.float)
        next_state = torch.tensor(t_next_state, dtype=torch.float)

        eval_q = self.eval_net.forward(state)
        target_q = self.target_net.forward(next_state)

        loss = torch.tensor(0., dtype=torch.float)
        for eq, a, r, tq, d in zip(eval_q, action, reward, target_q, done):
            qsa = eq[a]
            target = r + self.gamma * torch.max(tq) * (1-done)
            loss += (qsa - target)**2 / self.batch_size
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if done:
            torch.save(self.target_net.state_dict(), "./Table/task1-2.pt")

def train(env):
    agent = Agent(env)
    episode = 300
    rewards = []
    for ep in tqdm(range(episode)):
        state, info = env.reset()
        cnt = 0

        while True:
            agent.count += 1
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.buffer.insert(state, int(action), reward, next_state, int(terminated or truncated))
            cnt += reward
            state = next_state

            if len(agent.buffer) >= 1000:
                agent.update_table()
            if terminated or truncated:
                rewards.append(cnt)
                break

    total_reward.append(rewards)


def test(env):
    agent = Agent(env)
    agent.target_net.load_state_dict(torch.load("./Table/task1-2.pt"))
    rewards = []

    for _ in range(100):
        state, info = env.reset()
        cnt = 0
        while True:
            Q = agent.target_net.forward(
                torch.FloatTensor(state)).squeeze(0).detach()
            action = int(torch.argmax(Q).numpy())
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

    for i in range(3):
        print(f"## {i+1} training progress")
        train(env)
    
    test(env)

    env.close()

    os.makedirs("./Rewards", exist_ok=True)
    np.save("./Rewards/task1-2.npy", np.array(total_reward))
