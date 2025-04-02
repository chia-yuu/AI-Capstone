# cartpole, DQN
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
from collections import deque
import argparse
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_rewards = []
total_q_val = []
args = None

class replay_buffer():
    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    def __init__(self,  num_actions, hidden_layer_size=50):
        super(Net, self).__init__()
        self.input_state = 4  # the dimension of state space
        self.num_actions = num_actions  # the dimension of action space
        self.fc1 = nn.Linear(self.input_state, 32)  # input layer
        self.fc2 = nn.Linear(32, hidden_layer_size)  # hidden layer
        self.fc3 = nn.Linear(hidden_layer_size, num_actions)  # output layer

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class Agent():
    def __init__(self, env, epsilon=0.95, learning_rate=0.0002, GAMMA=0.97, batch_size=32, capacity=10000):
        self.env = env
        self.n_actions = 2  # the number of actions
        self.count = 0

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net(self.n_actions).to(device)
        self.target_net = Net(self.n_actions).to(device)

        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)  # Adam is a method using to optimize the neural network

    def learn(self):
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())

        # Begin your code
        # TODO
        # raise NotImplementedError("Not implemented yet.")
        # 2. Sample trajectories of batch size from the replay buffer.
        t_observations, actions, t_rewards, t_next_observations, done = self.buffer.sample(self.batch_size)

        observations = torch.tensor(np.array(t_observations), dtype=torch.float).to(device)
        rewards = torch.tensor(np.array(t_rewards), dtype=torch.float).to(device)
        next_observations = torch.tensor(np.array(t_next_observations), dtype=torch.float).to(device)

        # observations = torch.tensor(np.asarray(t_observations), dtype = torch.float)
        # rewards = torch.tensor(np.asarray(t_rewards).reshape(len(t_rewards), 1), dtype = torch.float)
        # next_observations = torch.tensor(np.asarray(t_next_observations), dtype = torch.float)

        # 3. Forward the data to the evaluate net and the target net.
        evaluate_q = self.evaluate_net.forward(observations)
        target_q   = self.target_net.forward(next_observations)

        # 4. Compute the loss with MSE.
        # loss = torch.tensor(0.,dtype=torch.float)
        # for eq, a, r, tq, d in zip(evaluate_q, actions, rewards, target_q, done):
        #     qsa = eq[a]     # Q(s, a)
        #     target = r + self.gamma * (1-d) * torch.max(tq)
        #     loss += (qsa - target)**2 / self.batch_size      # MSE
        
        q_values = evaluate_q.gather(1, torch.tensor(actions).unsqueeze(1).to(device))
        target_values = rewards + self.gamma * torch.max(target_q, dim=1)[0] * (1 - torch.tensor(done).to(device))
        loss = F.mse_loss(q_values, target_values.unsqueeze(1).to(device))

        # 5. Zero-out the gradients.
        self.optimizer.zero_grad()

        # 6. Backpropagation.
        loss.backward()

        # 7. Optimize the loss function.
        self.optimizer.step()

    def choose_action(self, state):
        with torch.no_grad():
            # Begin your code
            # TODO
            # raise NotImplementedError("Not implemented yet.")
            action = None
            if np.random.uniform(0, 1) > self.epsilon:
                # choose a random action
                action = env.action_space.sample()
            else:
                # choose the action with the highest Q
                s = torch.tensor(state, dtype = torch.float).to(device)
                q = self.evaluate_net(s)
                action = torch.argmax(q).item()
            # End your code
        return action

    def check_max_Q(self):
        # Begin your code
        # TODO
        # raise NotImplementedError("Not implemented yet.")
        state = self.env.reset()
        return torch.max(self.evaluate_net.forward(torch.tensor(state, dtype = torch.float)))
        # End your code


def train(env):
    agent = Agent(env)
    episode = 1000
    rewards = []
    action_record = [0] * 6
    for _ in tqdm(range(episode)):
        state, info = env.reset()
        count = 0
        while True:
            count += 1
            agent.count += 1
            # env.render()
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.buffer.insert(state, int(action), reward, next_state, int(done))
            action_record[action] += 1
            
            if len(agent.buffer) >= 1000:
                agent.learn()
            if done:
                rewards.append(count)
                break
            state = next_state
    torch.save(agent.target_net.state_dict(), "./Table/task2-2.pt")
    total_rewards.append(rewards)
    print(f"actions: {action_record}, ", end=' ')
    print(f"Average reward = {np.mean(rewards)}")


def test(env):
    rewards = []
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(torch.load("./Table/task2-2.pt", map_location=device))
    for _ in tqdm(range(100)):
        state, info = env.reset()
        count = 0
        while True:
            if args.visualize:
                env.render()
            count += 1
            Q = testing_agent.target_net.forward(torch.tensor(np.array(state), dtype=torch.float).to(device)).squeeze(0).detach()
            # action = int(torch.argmax(Q).numpy())
            action = int(torch.argmax(Q).cpu().numpy())
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done:
                rewards.append(count)
                break
            state = next_state

    print(f"reward: {np.mean(rewards)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testOnly", action='store_true')
    parser.add_argument("--visualize", action='store_true')
    args = parser.parse_args()

    print(f"## Using {device} to train ##")
    env = gym.make('CartPole-v1')
    if args.visualize:
        env = gym.make('CartPole-v1', render_mode='human')  # visualize
    os.makedirs("./Table", exist_ok=True)
    os.makedirs("./Rewards", exist_ok=True)
    os.makedirs("./Q Values", exist_ok=True)

    if not args.testOnly:
        for i in range(5):
            print(f"## {i+1} training progress")
            train(env)
            np.save("./Rewards/task2-2.npy", np.array(total_rewards))
    
    print("## testing progress")
    test(env)

    env.close()