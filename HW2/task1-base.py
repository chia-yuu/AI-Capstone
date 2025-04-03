# space invader, base line
import gymnasium as gym
import ale_py
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
import random

total_reward = []
args = None

def train(env):
    episode = 1000
    rewards = []
    action_record = [0, 0, 0, 0, 0, 0]
    for ep in tqdm(range(episode)):
        state, info = env.reset()
        r_cnt = 0
        cnt = 0

        while True:
            if args.visualize:
                env.render()
            cnt += 1
            action = random.choice([0,1,2,3,4,5])
            n_state, reward, terminated, truncated, info = env.step(action)
            r_cnt += reward
            action_record[action] += 1

            if terminated or truncated:
                rewards.append(r_cnt)
                break
    total_reward.append(rewards)
    print(f"actions: {action_record}, ", end=" ")
    print(f"Average reward = {np.mean(rewards)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--testOnly", action='store_true')
    args = parser.parse_args()

    gym.register_envs(ale_py)
    env = gym.make("ALE/SpaceInvaders-v5")
    if args.visualize:
        env = gym.make("ALE/SpaceInvaders-v5", render_mode='human')  # visualize
    os.makedirs("./Table", exist_ok=True)
    os.makedirs("./Rewards", exist_ok=True)
    os.makedirs("./Q Values", exist_ok=True)

    if not args.testOnly:
        for i in range(5):
            print(f"## {i+1} training progress")
            train(env)
            np.save("./Rewards/task1-base.npy", np.array(total_reward))

    env.close()
