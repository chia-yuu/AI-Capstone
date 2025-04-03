import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import seaborn as sns
from matplotlib.patches import Patch

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
    # plot reward
    # reward = np.load("./Rewards/task3-1.npy").transpose()
    reward = np.load("./Rewards/task3-1.npy")[0, :]
    avg = (
        np.convolve(
            np.array(reward).flatten(), np.ones(int(epoch//500)), mode="valid"
        )
        / int(epoch//500)
        )
    # avg = np.mean(reward, axis=1)
    # std = np.std(reward, axis=1)
    initialize_plot("Blackjack-v1 Q Learning")
    plt.plot([i for i in range(len(avg))], avg,
             label='Blackjack', color='orange')
    # plt.fill_between([i for i in range(epoch)],
    #                  avg+std, avg-std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Plots/task3-1.png")
    plt.show()
    plt.close()

    # plot q value
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

    # plot grid
    q_table = np.load("./Table/task3-1.npy")
    q = np.max(q_table[:, :, :, 0], axis=2)       # q value
    a = np.argmax(q_table[:, :, :, 0], axis=2)    # action
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    x, y = np.meshgrid(np.arange(q.shape[1]), np.arange(q.shape[0]))

    ax1.plot_surface(x, y, q, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('Dealer show')
    ax1.set_ylabel('Player sum')
    ax1.set_zlabel('Value')
    ax1.set_title("State values: with usable ace")

    ax2 = fig.add_subplot(122)
    ax2 = sns.heatmap(a[4:22, 1:], linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_xlabel('Dealer show')
    ax2.set_ylabel('Player sum')
    ax2.set_title("Actions: with usable ace")
    ax2.set_yticklabels(range(4, 22))
    ax2.set_xticklabels(["A"] + list(range(2, 11)), fontsize=12)
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))

    plt.tight_layout()
    plt.savefig("./Plots/task3-1-00.png")
    plt.show()
    plt.close()

    q = np.max(q_table[:, :, :, 1], axis=2)       # q value
    a = np.argmax(q_table[:, :, :, 1], axis=2)    # action
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    x, y = np.meshgrid(np.arange(q.shape[1]), np.arange(q.shape[0]))

    ax1.plot_surface(x, y, q, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('Dealer show')
    ax1.set_ylabel('Player sum')
    ax1.set_zlabel('Value')
    ax1.set_title("State values: without usable ace")

    ax2 = fig.add_subplot(122)
    ax2 = sns.heatmap(a[4:22, 1:], linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_xlabel('Dealer show')
    ax2.set_ylabel('Player sum')
    ax2.set_title("Actions: without usable ace")
    ax2.set_yticklabels(range(4, 22))
    ax2.set_xticklabels(["A"] + list(range(2, 11)), fontsize=12)
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))

    plt.tight_layout()
    plt.savefig("./Plots/task3-1-11.png")
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
