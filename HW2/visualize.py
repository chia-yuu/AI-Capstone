import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def initialize_plot(title):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('rewards')

def task1_1(epoch):
    reward = np.load("./Rewards/task1-1.npy").transpose()
    avg = np.mean(reward, axis=1)
    std = np.std(reward, axis=1)
    initialize_plot("SpaceInvaders-v5 Using Q Learning")
    plt.plot([i for i in range(epoch)], avg,
             label='space_invader', color='orange')
    plt.fill_between([i for i in range(epoch)],
                     avg+std, avg-std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Plots/task1-1.png")
    plt.show()
    plt.close()

def task1_2(epoch):
    reward = np.load("./Rewards/task1-2.npy").transpose()
    print(reward.shape)
    avg = np.mean(reward, axis=1)
    std = np.std(reward, axis=1)
    initialize_plot("SpaceInvaders-v5 DQN")
    plt.plot([i for i in range(epoch)], avg,
             label='space_invader', color='orange')
    plt.fill_between([i for i in range(epoch)],
                     avg+std, avg-std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Plots/task1-2.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1_1", action="store_true")
    parser.add_argument("--task1_2", action="store_true")
    parser.add_argument("--episode", type=int, default=3000)
    args = parser.parse_args()

    os.makedirs("./Plots", exist_ok=True)

    if args.task1_1:
        task1_1(args.episode)
    elif args.task1_2:
        task1_2(args.episode)
