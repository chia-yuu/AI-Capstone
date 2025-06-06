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
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_reward = []
total_q_val = []
args = None

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
        # self.input_state = 210 * 160 * 3    # observation space
        # self.n_action = n_action            # action space
        # self.fc1 = nn.Linear(self.input_state, 32)
        # self.fc2 = nn.Linear(32, hidden_layer_size)
        # self.fc3 = nn.Linear(hidden_layer_size, n_action)

        self.channel = 3
        self.height = 210
        self.width = 160
        self.n_action = n_action
        self.conv1 = nn.Conv2d(self.channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(22528, 512)  # Adjusted input size for the fully connected layer
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_action)

    def forward(self, state):
        # state = state.view(state.size(0), -1)
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # q = self.fc3(x)
        # return q
        if(state.dim() == 3):
            state = state.unsqueeze(0)
        state = state.permute(0, 3, 1, 2)   # [1, 210, 160, 3] to [1, 3, 210, 160]
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output
        x = self.flatten(x)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output layer
        q = self.fc3(x)
        
        return q


# Agent to perform DQN
class Agent():
    def __init__(self, env, epsilon=0.05, lr=0.0002, gamma=0.9, batch_size=32, capacity=10000):
        self.env = env
        self.epsilon = epsilon  # explore or exploit
        self.lr = lr
        self.gamma = gamma

        self.n_action = 6
        self.count = 0      # when to update net
        self.batch_size = batch_size
        self.capacity = capacity
        
        self.buffer = Replay_buffer(self.capacity)
        self.eval_net = Model(self.n_action).to(device)
        self.target_net = Model(self.n_action).to(device)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
    
    # choose an action with the given state
    def choose_action(self, state):
        q_mean = 0
        with torch.no_grad():
            action = None
            if np.random.uniform(0, 1) < self.epsilon:
                # random action
                action = env.action_space.sample()
            else:
                # action with max Q
                s = torch.tensor(state, dtype=torch.float).to(device)
                q = self.eval_net(s)
                # action = torch.argmax(q).item()
                action = q.sum(dim=0).argmax().item()
                q_mean = q.mean().item()
        return action, q_mean
    
    # update Q table based on reward & state after taking action
    def update_table(self):
        if self.count%100 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        t_state, action, t_reward, t_next_state, done = self.buffer.sample(self.batch_size)
        # state = torch.tensor(t_state, dtype=torch.float)
        # reward = torch.tensor(t_reward, dtype=torch.float)
        # next_state = torch.tensor(t_next_state, dtype=torch.float)
        state = torch.tensor(np.array(t_state), dtype=torch.float).to(device)
        reward = torch.tensor(np.array(t_reward), dtype=torch.float).to(device)
        next_state = torch.tensor(np.array(t_next_state), dtype=torch.float).to(device)

        eval_q = self.eval_net.forward(state)
        target_q = self.target_net.forward(next_state)

        # loss = torch.tensor(0., dtype=torch.float)
        # for eq, a, r, tq, d in zip(eval_q, action, reward, target_q, done):
        #     qsa = eq[a]
        #     target = r + self.gamma * torch.max(tq) * (1-d)
        #     loss += (qsa - target)**2 / self.batch_size
        q_values = eval_q.gather(1, torch.tensor(action).unsqueeze(1).to(device))
        target_values = reward + self.gamma * torch.max(target_q, dim=1)[0] * (1 - torch.tensor(done).to(device))
        loss = F.mse_loss(q_values, target_values.unsqueeze(1).to(device))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # if done:
        #     torch.save(self.target_net.state_dict(), "./Table/task1-2.pt")

def train(env):
    agent = Agent(env)
    episode = 1000
    rewards = []
    q_val = []
    action_record = [0] * 6
    for ep in tqdm(range(episode)):
        state, info = env.reset()
        r_cnt = 0
        q_cnt = 0
        cnt = 0

        while True:
            cnt += 1
            agent.count += 1
            action, q_mean = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.buffer.insert(np.array(state), int(action), reward, np.array(next_state), int(terminated or truncated))
            r_cnt += reward
            state = next_state
            q_cnt += q_mean
            action_record[action] += 1

            if len(agent.buffer) >= 100:
                agent.update_table()
            if terminated or truncated:
                rewards.append(r_cnt)
                q_val.append(q_cnt / cnt)
                if (ep + 1)%200 == 0:
                    torch.save(agent.target_net.state_dict(), f"./Table/task1-2-{ep}.pt")
                break
    torch.save(agent.target_net.state_dict(), "./Table/task1-2.pt")
    total_reward.append(rewards)
    total_q_val.append(q_val)
    print(f"actions: {action_record}, ", end=' ')
    print(f"Average reward = {np.mean(rewards)}")


def test(env):
    agent = Agent(env)
    agent.target_net.load_state_dict(torch.load("./Table/task1-2.pt", map_location=device))
    rewards = []
    action_record = [0] * 6
    mx = 0
    best_record = None

    for ep in tqdm(range(100)):
        state, info = agent.env.reset()
        cnt = 0
        while True:
            if args.visualize:
                agent.env.render()    # visualize
            Q = agent.target_net.forward(torch.tensor(np.array(state), dtype=torch.float).to(device)).squeeze(0).detach()
            # action = int(torch.argmax(Q).numpy())
            action = int(torch.argmax(Q).cpu().numpy())
            next_state, reward, terminated, truncated, info = agent.env.step(action)
            cnt += reward
            state = next_state
            action_record[action] += 1

            if terminated or truncated:
                rewards.append(cnt)
                break
        if (ep+1)%10 == 0:
            avg = np.mean(rewards)
            if avg > mx:
                mx = avg
                best_record = action_record
    print(f"actions: {best_record}, ", end=' ')
    print(f"Average reward = {mx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testOnly", action='store_true')
    parser.add_argument("--visualize", action='store_true')
    args = parser.parse_args()

    print(f"## Using {device} to train ##")
    gym.register_envs(ale_py)
    env = gym.make("ALE/SpaceInvaders-v5")
    if args.visualize:
        env = gym.make("ALE/SpaceInvaders-v5", render_mode='human')  # visualize
    os.makedirs("./Table", exist_ok=True)
    os.makedirs("./Rewards", exist_ok=True)

    if not args.testOnly:
        train(env)
        np.save("./Rewards/task1-2.npy", np.array(total_reward))
        np.save("./Q Values/task1-2.npy", np.array(total_q_val))
    
    print("## Testing progress")
    test(env)

    env.close()
