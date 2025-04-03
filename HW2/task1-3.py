# space invader, DQN with frame stack & normalize
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
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_reward = []
total_q_val = []
stack_size = 4
args = None

# Frame preprocessing function
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cropped_frame = gray[8:-12, 4:-12]  # Crop screen
    normalized_frame = cropped_frame / 255.0  # Normalize pixel values
    preprocessed_frame = cv2.resize(normalized_frame, (84, 110))  # Resize frame
    return preprocessed_frame  # Return processed frame

# Stack frames for temporal awareness
def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        stacked_frames = deque([np.zeros((110, 84), dtype=np.float32) for _ in range(stack_size)], maxlen=4)
        for _ in range(stack_size):
            stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)
    stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames

# Experience replay buffer
class ReplayBuffer():
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
    
# Deep Q-Network
class Model(nn.Module):
    def __init__(self, n_action):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(stack_size, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4480, 512)  # Adjusted input size for the fully connected layer
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_action)
    
    def forward(self, state):
        state = state.to(device)
        if(state.dim() == 3):
            state = state.unsqueeze(0)
        state = state.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

class Agent():
    def __init__(self, env, epsilon=0.05, lr=0.0002, gamma=0.9, batch_size=32, capacity=10000):
        self.env = env
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.n_action = env.action_space.n
        self.count = 0
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(capacity)
        self.eval_net = Model(self.n_action).to(device)
        self.target_net = Model(self.n_action).to(device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
    
    def choose_action(self, state):
        q_mean = 0
        with torch.no_grad():
            action = None
            if np.random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space.sample()
            else:
                s = torch.tensor(state, dtype=torch.float).to(device)
                q = self.eval_net(s)
                action = q.sum(dim=0).argmax().item()
                q_mean = q.mean().item()
        return action, q_mean
    
    def update_table(self):
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        
        state = torch.tensor(np.array(state), dtype=torch.float32).to(device)
        reward = torch.tensor(np.array(reward), dtype=torch.float32).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).to(device)
        
        eval_q = self.eval_net.forward(state)
        target_q = self.target_net.forward(next_state)
        
        q_values = eval_q.gather(1, torch.tensor(action).unsqueeze(1).to(device))
        target_values = reward + self.gamma * torch.max(target_q, dim=1)[0] * (1 - torch.tensor(done).to(device))
        loss = F.mse_loss(q_values, target_values.unsqueeze(1).to(device))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train(env):
    agent = Agent(env)
    episodes = 50
    rewards = []
    q_val = []
    action_record = [0, 0, 0, 0, 0, 0]
    stacked_frames = deque([np.zeros((110, 84), dtype=np.float32) for _ in range(stack_size)], maxlen=4)
    
    for ep in tqdm(range(episodes)):
        state, info = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        r_cnt = 0
        q_cnt = 0
        cnt = 0
        
        while True:
            agent.count += 1
            cnt += 1
            action, q_mean = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            agent.buffer.insert(state, action, reward, next_state, int(terminated or truncated))
            r_cnt += reward
            q_cnt += q_mean
            state = next_state
            action_record[action] += 1
            
            if len(agent.buffer) >= 100:
                agent.update_table()
            
            if terminated or truncated:
                rewards.append(r_cnt)
                q_val.append(q_cnt / cnt)
                break
    torch.save(agent.target_net.state_dict(), "./Table/task1-3.pt")
    total_reward.append(rewards)
    total_q_val.append(q_val)
    print(f"actions: {action_record}, ", end=' ')
    print(f"Average reward = {np.mean(rewards)}")

def test(env):
    agent = Agent(env)
    agent.target_net.load_state_dict(torch.load("./Table/task1-3.pt"))
    rewards = []
    action_record = [0, 0, 0, 0, 0, 0]
    stacked_frames = deque([np.zeros((110, 84), dtype=np.float32) for _ in range(stack_size)], maxlen=4)
    
    for _ in tqdm(range(100)):
        state, info = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        cnt = 0
        
        while True:
            if args.visualize:
                agent.env.render()    # visualize
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action = torch.argmax(agent.target_net(state_tensor)).item()
            next_state, reward, terminated, truncated, info = agent.env.step(action)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            cnt += reward
            state = next_state
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

    print(f"## Using {device} to train ##")
    gym.register_envs(ale_py)
    env = gym.make("ALE/SpaceInvaders-v5")
    if args.visualize:
        env = gym.make("ALE/SpaceInvaders-v5", render_mode='human')  # visualize
    os.makedirs("./Table", exist_ok=True)
    os.makedirs("./Rewards", exist_ok=True)

    if not args.testOnly:
        train(env)
        np.save("./Rewards/task1-3.npy", np.array(total_reward))
        np.save("./Q Values/task1-3.npy", np.array(total_q_val))
    
    print("## Testing progress")
    test(env)

    env.close()