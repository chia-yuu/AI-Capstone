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
    initialize_plot("SpaceInvaders-v5 Q Learning Reward")
    plt.plot([i for i in range(epoch)], avg, label='space_invader', color='orange')
    plt.fill_between([i for i in range(epoch)], avg+std, avg-std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Plots/task1-1.png")
    plt.show()
    plt.close()

    q_val = np.load("./Q Values/task1-1.npy").transpose()
    # q_val = q_val.reshape(60, 5)
    avg = np.mean(q_val, axis=1)
    std = np.std(q_val, axis=1)
    initialize_plot("SpaceInvaders-v5 Q Learning Q Valuse")
    plt.plot([i for i in range(int(epoch//5))], avg, label='space_invader', color='orange')
    plt.fill_between([i for i in range(int(epoch//5))], avg+std, avg-std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Plots/task1-1-q.png")
    plt.show()
    plt.close()

def task1_2(epoch):
    reward = np.load("./Rewards/task1-2.npy").transpose()
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

def task1_3(epoch):
    reward = np.load("./Rewards/task1-3.npy").transpose()
    avg = np.mean(reward, axis=1)
    std = np.std(reward, axis=1)
    initialize_plot("SpaceInvaders-v5 DQN with stack frame")
    plt.plot([i for i in range(epoch)], avg,
             label='space_invader', color='orange')
    plt.fill_between([i for i in range(epoch)],
                     avg+std, avg-std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Plots/task1-3.png")
    plt.show()
    plt.close()

    q_val = np.load("./Q Values/task1-3.npy").transpose()
    avg = np.mean(q_val, axis=1)
    std = np.std(q_val, axis=1)
    initialize_plot("SpaceInvaders-v5 Q Learning Q Values")
    plt.plot([i for i in range(int(epoch))], avg, label='space_invader', color='orange')
    plt.fill_between([i for i in range(int(epoch))], avg+std, avg-std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Plots/task1-3-q.png")
    plt.show()
    plt.close()

def task1_compare(epoch):
    reward1 = np.load("./Rewards/task1-1.npy").transpose()
    avg1 = np.mean(reward1, axis=1)

    reward2 = np.load("./Rewards/task1-2.npy").transpose()
    avg2 = np.mean(reward2, axis=1)

    reward3 = np.load("./Rewards/task1-3.npy").transpose()
    avg3 = np.mean(reward3, axis=1)

    initialize_plot("SpaceInvaders-v5 Reward Compare")
    plt.plot([i for i in range(300)], avg1, label='task1-1', color='green', alpha=0.5)
    plt.plot([i for i in range(1000)], avg2, label='task1-2', color='orange', alpha=0.5)
    plt.plot([i for i in range(1000)], avg3, label='task1-3', color='yellow', alpha=0.5)
    plt.legend(loc="best")
    plt.savefig("./Plots/task1 compare.png")
    plt.show()
    plt.close()

def task2_1(epoch):
    reward = np.load("./Rewards/task2-1.npy").transpose()
    avg = np.mean(reward, axis=1)
    std = np.std(reward, axis=1)
    initialize_plot("CartPole-v1 Q Learning")
    plt.plot([i for i in range(epoch)], avg,
             label='cartpole', color='orange')
    plt.fill_between([i for i in range(epoch)],
                     avg+std, avg-std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Plots/task2-1.png")
    plt.show()
    plt.close()

    q_val = np.load("./Q Values/task2-1.npy").transpose()
    avg = np.mean(q_val, axis=1)
    std = np.std(q_val, axis=1)
    initialize_plot("CartPole-v1 Q Learning Q Values")
    plt.plot([i for i in range(int(epoch//5))], avg, label='space_invader', color='orange')
    plt.fill_between([i for i in range(int(epoch//5))], avg+std, avg-std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Plots/task2-1-q.png")
    plt.show()
    plt.close()

def task2_2(epoch):
    reward = np.load("./Rewards/task2-2.npy").transpose()
    avg = np.mean(reward, axis=1)
    std = np.std(reward, axis=1)
    initialize_plot("CartPole-v1 DQN")
    plt.plot([i for i in range(epoch)], avg,
             label='cartpole', color='orange')
    plt.fill_between([i for i in range(epoch)],
                     avg+std, avg-std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Plots/task2-2.png")
    plt.show()
    plt.close()

def task3_1(epoch):
    reward = np.load("./Rewards/task3-1.npy").transpose()
    print(reward.shape)
    new_reward = reward[::3000, :]
    print(new_reward.shape)
    avg = np.mean(new_reward, axis=1)
    std = np.std(new_reward, axis=1)
    initialize_plot("Blackjack-v1 Q Learning")
    plt.plot([i for i in range(1000)], avg,
             label='Blackjack', color='orange')
    plt.fill_between([i for i in range(1000)],
                     avg+std, avg-std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Plots/task3-1.png")
    plt.show()
    plt.close()

    q_val = np.load("./Q Values/task3-1.npy").transpose()
    avg = np.mean(q_val, axis=1)
    std = np.std(q_val, axis=1)
    initialize_plot("Blackjack-v1 Q Learning Q Values")
    plt.plot([i for i in range(int(epoch//5))], avg, label='Blackjack', color='orange')
    plt.fill_between([i for i in range(int(epoch//5))], avg+std, avg-std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Plots/task3-1-q.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1_1", action="store_true")
    parser.add_argument("--task1_2", action="store_true")
    parser.add_argument("--task1_3", action="store_true")
    parser.add_argument("--task2_1", action="store_true")
    parser.add_argument("--task2_2", action="store_true")
    parser.add_argument("--task3_1", action="store_true")
    parser.add_argument("--task1-compare", action="store_true")
    parser.add_argument("--episode", type=int, default=300)
    args = parser.parse_args()

    os.makedirs("./Plots", exist_ok=True)

    if args.task1_1:
        task1_1(args.episode)
    elif args.task1_2:
        task1_2(args.episode)
    elif args.task1_3:
        task1_3(args.episode)
    elif args.task1_compare:
        task1_compare(args.episode)
    elif args.task2_1:
        task2_1(args.episode)
    elif args.task2_2:
        task2_2(args.episode)
    elif args.task3_1:
        task3_1(args.episode)
