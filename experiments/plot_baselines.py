import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.plotting_utils import get_scores, generate_plot, get_all_run_titles, get_all_runs_and_subruns, get_plot_params
import os
import re
import numpy as np

directory = "baseline_results/SAC/"
experiment_name = "SAC_20"
label = "lunar_lander"

def main():
    regex = re.compile(directory + experiment_name + ".*")
    path_list = []
    for dirname, _, _ in os.walk(directory):
        if regex.match(dirname):
            path_list.append(dirname + "/progress.csv")
    plot_curve(path_list)


def plot_curve(path_list):
    episode_data = []
    reward_data = []
    for path in path_list:
        df = pd.read_csv(path, sep=",")
        max_index = df.loc[df['time/episodes'] == 1000].index[0]  # Can change this part to match specific files
        episode_list = []
        reward_list = []
        for i in range(max_index + 1):
            episode = df.loc[i]["time/episodes"]
            reward = df.loc[i]["rollout/ep_rew_mean"]
            episode_list.append(episode)
            reward_list.append(reward)
        episode_data.append(episode_list)
        reward_data.append(reward_list)
    median, means, top, bot = get_plot_params(np.array(reward_data))
    plt.plot(range(len(means)), means, linewidth=2, alpha=0.9, label=label)
    plt.fill_between(range(len(means)), top, bot, alpha=0.2)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()