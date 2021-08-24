import numpy as np
import pickle
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from pathlib import Path

from common.plotting_utils import generate_plot, get_all_run_titles, get_all_runs_and_subruns
import pdb

def get_loss(log_dir,
               subdir="scores",
               only_longest=False,
               min_length=-1,
               cumulative=False):
    longest = 0
    print(log_dir)
    overall_scores = []
    glob_pattern = log_dir  + "/loss" + "*.txt"
    for score_file in glob.glob(glob_pattern):
        with open(score_file, "rb") as f:
            scores = []
            for x in f:
                scores.append(float(x))
        if only_longest:
            if len(scores) > longest:
                overall_scores = [scores]
                longest = len(scores)
        else:
            if len(scores) >= min_length:  # -1 always passes!
                overall_scores.append(scores)

    if len(overall_scores) == 0:
        raise Exception("No scores in " + log_dir)

    min_length = min(len(s) for s in overall_scores)

    overall_scores = [s[:min_length] for s in overall_scores]

    score_array = np.array(overall_scores)
    # print(score_array.shape)
    if cumulative:
        score_array = np.cumsum(score_array, axis=1)
    return score_array

def make_graphs(experiment_name,
                subdir,
                run_titles=None,
                smoothen=False,
                min_length=-1,
                only_longest=False,
                skip_failures=False,
                cumulative=False,
                all_seeds=False):

    if run_titles is None:
        print("Using all runs in experiment")
        run_titles = get_all_run_titles(experiment_name=experiment_name)

    log_dirs = [
        os.path.join(experiment_name, run_title) for run_title in run_titles
    ]

    score_arrays = []
    good_run_titles = []
    
    for log_dir, run_title in zip(log_dirs, run_titles):
        try:
            scores = get_loss(log_dir,
                                subdir=subdir,
                                only_longest=only_longest,
                                min_length=min_length,
                                cumulative=cumulative)
            
            score_arrays.append(scores)
            good_run_titles.append(run_title)

            if all_seeds:
                for i, score in enumerate(scores):
                    score_arrays.append(np.array([score]))
                    good_run_titles.append(run_title + f"_{i+1}")

        except Exception as e:
            print(f"skipping {log_dir} due to error {e}")
            pass

    [
        generate_plot(score_array, run_title, smoothen=smoothen)
        for score_array, run_title in zip(score_arrays, good_run_titles)
    ]

    plt.ylabel(subdir.replace("_", " "))
    plt.xlabel("Episode")
    plt.legend()

    plt.show()


def main():
    """
    Change these options and directories to suit your needs
    """
    ## Defaults
    subdir = "average_loss"
    smoothen = False
    min_length = -1
    only_longest = False
    cumulative = False
    all_seeds = False

    ## Options (Uncomment as needed)
    # smoothen = True
    # min_length = 300
    # only_longest = True
    # cumulative = True
    # all_seeds = True

    experiment_name = "./results/BipedalWalker"
    run_titles = get_all_run_titles(experiment_name)
    make_graphs(experiment_name,
                subdir,
                run_titles=run_titles,
                smoothen=smoothen,
                min_length=min_length,
                only_longest=only_longest,
                cumulative=cumulative,
                all_seeds=all_seeds)


if __name__ == '__main__':
    main()