# cartpole, q learning
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt

total_reward = []
total_q_val = []
episode = 3000
decay = 0.045
args = None

class Agent():
    def __init__(self, env, epsilon=0.95, learning_rate=0.5, GAMMA=0.97, num_bins=7):
        self.env = env

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA

        self.num_bins = num_bins
        self.qtable = np.zeros((self.num_bins, self.num_bins,
                               self.num_bins, self.num_bins, self.env.action_space.n))

        # init_bins() is your work to implement.
        self.bins = [
            self.init_bins(-2.4, 2.4, self.num_bins),  # cart position
            self.init_bins(-3.0, 3.0, self.num_bins),  # cart velocity
            self.init_bins(-0.5, 0.5, self.num_bins),  # pole angle
            self.init_bins(-2.0, 2.0, self.num_bins)  # tip velocity
        ]

    def init_bins(self, lower_bound, upper_bound, num_bins):
        # Begin your code
        # TODO
        # raise NotImplementedError("Not implemented yet.")
        # print(np.linspace(lower_bound, upper_bound, num_bins))                # [0, x, x, x, x, 10]
        ret = np.linspace(lower_bound, upper_bound, num_bins,endpoint=False)    # [0, 2, 4, 6, 8]
        return ret[1:-1]                                                        # [2, 4, 6, 8]
        # End your code

    def discretize_value(self, value, bins):
        # Begin your code
        # TODO
        # raise NotImplementedError("Not implemented yet.")
        return np.digitize(value, bins)
        # End your code

    def discretize_observation(self, observation):
        # Begin your code
        # TODO
        # raise NotImplementedError("Not implemented yet.")
        # print(observation)
        ret = [0] * 4
        ret[0] = self.discretize_value(observation[0], self.bins[0])
        ret[1] = self.discretize_value(observation[1], self.bins[1])
        ret[2] = self.discretize_value(observation[2], self.bins[2])
        ret[3] = self.discretize_value(observation[3], self.bins[3])
        return ret
        # End your code

    def choose_action(self, state):
        # Begin your code
        # TODO
        # raise NotImplementedError("Not implemented yet.")
        action = None
        if np.random.uniform(0, 1) > self.epsilon:
            # choose a random action
            action = env.action_space.sample()
        else:
            # choose the action with the highest Q
            action = np.argmax(self.qtable[state[0], state[1], state[2], state[3], :])
            # print("q table:"), print(self.qtable[state[0], state[1], state[2], state[3], :])
            # print("action:"), print(action)
        return action
        # End your code

    def learn(self, state, action, reward, next_state, done):
        # Begin your code
        # TODO
        # raise NotImplementedError("Not implemented yet.")
        qsa = self.qtable[state[0], state[1], state[2], state[3], action]
        mx = np.max(self.qtable[next_state[0], next_state[1], next_state[2], next_state[3], :])
        if done: mx = 0
        self.qtable[state[0], state[1], state[2], state[3], action] = qsa + self.learning_rate * (reward + self.gamma * mx - qsa)
        if done:
        # End your code
            np.save("./Table/task2-1.npy", self.qtable)

    def check_max_Q(self):
        # Begin your code
        # TODO
        # raise NotImplementedError("Not implemented yet.")
        state = self.discretize_observation(self.env.reset())
        return np.max(self.qtable[state[0], state[1], state[2], state[3], :])
        # End your code


def train(env):
    training_agent = Agent(env)
    rewards = []
    q_val = []
    action_record = [0] * 6
    for ep in tqdm(range(episode)):
        state, _ = env.reset()
        state = training_agent.discretize_observation(state)
        done = False

        count = 0
        while True:
            count += 1
            action = training_agent.choose_action(state)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = training_agent.discretize_observation(next_observation)
            training_agent.learn(state, action, reward, next_state, done)

            if done:
                rewards.append(count)
                break
            state = next_state
            action_record[action] += 1

        if (ep + 1) % 500 == 0:
            training_agent.learning_rate -= decay
        if (ep + 1) % 5 == 0:
            q_val.append(training_agent.qtable.mean())

        np.save("./Table/task2-1.npy", training_agent.qtable)
    total_reward.append(rewards)
    total_q_val.append(q_val)
    print(f"actions: {action_record}, ", end=' ')
    print(f"Average reward = {np.mean(rewards)}")


def test(env):
    testing_agent = Agent(env)

    testing_agent.qtable = np.load("./Table/task2-1.npy")
    rewards = []
    action_record = [0] * 6

    for _ in range(100):
        state, _ = testing_agent.env.reset()
        state = testing_agent.discretize_observation(state)
        count = 0
        while True:
            if args.visualize:
                env.render()
            count += 1
            action = np.argmax(testing_agent.qtable[tuple(state)])
            next_observation, _, terminated, truncated, _ = testing_agent.env.step(action)
            done = terminated or truncated

            if done == True:
                rewards.append(count)
                break

            next_state = testing_agent.discretize_observation(next_observation)
            state = next_state
            action_record[action] += 1

    print(f"actions: {action_record}, ", end=' ')
    print(f"Average reward = {np.mean(rewards)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testOnly", action='store_true')
    parser.add_argument("--visualize", action='store_true')
    args = parser.parse_args()

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
            np.save("./Rewards/task2-1.npy", np.array(total_reward))
            np.save("./Q Values/task2-1.npy", np.array(total_q_val))
    
    print("## testing progress")
    test(env)

    env.close()
