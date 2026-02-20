import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

def plot_blackjack_values(V):
    def get_Z(x, y, usable_ace):
        return V.get((x, y, usable_ace), 0)

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([get_Z(x, y, usable_ace) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel("Player's Current Sum")
        ax.set_ylabel("Dealer's Showing Card")
        ax.set_zlabel("State Value")
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()
    plt.close()

def plot_policy(policy):
    def get_Z(x, y, usable_ace):
        return policy.get((x, y, usable_ace), 1)  # default to HIT

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(10, 0, -1)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(x, y, usable_ace) for x in x_range] for y in y_range])
        surf = ax.imshow(Z, cmap=plt.get_cmap('Pastel2', 2), vmin=0, vmax=1, extent=[10.5, 21.5, 0.5, 10.5])
        ax.set_xticks(x_range)
        ax.set_yticks(y_range)
        ax.set_xlabel("Player's Current Sum")
        ax.set_ylabel("Dealer's Showing Card")
        ax.grid(color='w', linestyle='-', linewidth=1)
        plt.gca().invert_yaxis()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, ticks=[0, 1], cax=cax)
        cbar.ax.set_yticklabels(['0 (STICK)', '1 (HIT)'])

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121)
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(122)
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()
    plt.close()

def plot_q_heatmap(Q, state_filter_fn):
    """Plot Q[state][action] for filtered states."""
    import seaborn as sns
    states = [s for s in Q.keys() if state_filter_fn(s)]
    hit_vals = [Q[s][1] for s in states]
    stick_vals = [Q[s][0] for s in states]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(hit_vals, color="red", label="HIT", kde=True, stat="density", ax=ax)
    sns.histplot(stick_vals, color="blue", label="STICK", kde=True, stat="density", ax=ax)
    ax.set_title("Q-Value Distribution for Filtered States")
    ax.set_xlabel("Q(s,a)")
    ax.legend()
    plt.show()
    plt.close()

def plot_rewards(reward_list, window=1000):
    """Plot reward per episode (moving average)."""
    rewards = np.array(reward_list)
    means = np.convolve(rewards, np.ones(window) / window, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(means)
    plt.title(f"Average Reward (Window = {window})")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.show()
    plt.close()