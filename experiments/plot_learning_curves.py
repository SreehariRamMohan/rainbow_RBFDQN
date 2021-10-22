import numpy as np
import pickle
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from pathlib import Path
import matplotlib.ticker as ticker
from matplotlib.pyplot import figure

'''
Onager Prelaunch command
onager prelaunch +jobname distributional_ant_sweep_location_lr +command "python -u experiments/experiment.py --hyper_parameter_name 50 --experiment_name ./results/Ant --reward_norm clip --distributional True --learning_rate 1e-5 --vmin 0.0 --vmax 500.0 --numpoints 130 --num_atoms 101" +arg --learning_rate_location_side 1e-5 1e-4 1e-3 1e-2 1e-1 +arg --seed 0 +tag --run_title +tag-args --learning_rate_location_side --seed
'''

from common.plotting_utils import get_scores, generate_plot, get_all_run_titles, get_all_runs_and_subruns


def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def make_graphs(experiment_name,
                subdir,
                run_titles=None,
                smoothen=False,
                min_length=-1,
                only_longest=False,
                skip_failures=False,
                cumulative=False,
                all_seeds=False,
                use_onager=False):

    if run_titles is None:
        print("Using all runs in experiment")
        run_titles = get_all_run_titles(experiment_name=experiment_name)

    run_titles = list(filter(lambda x: not "*" in x, run_titles))

    log_dirs = [
        os.path.join(experiment_name, run_title) for run_title in run_titles
    ]

    score_arrays = []
    good_run_titles = []
    for log_dir, run_title in zip(log_dirs, run_titles):
        try:
            scores = get_scores(log_dir,
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
    
    if use_onager:
        '''
        When performing onager sweeps you can end up with hundreds and hundreds of variations. 
        Plotting all of this and manually combing through can be messy. 
        This short snippet of code can prune out the terrible performing runs and just give you the top N_BEST performing ones. 
        '''
        # the number of "top runs" you want to plot
        N_BEST = 5
        
        # the length of the trajectory (from the end) you want to use as a heuristic for ranking all the runs 
        RUN_HEURISTIC = 20

        # the minimum length of a trajectory to be considered in our ranking. Set this to something positive so the code 
        # won't crash if it tries to plot a run which is 0 in length. 
        MIN_LENGTH = 100

        # only works for reward plotting for now
        if "reward" not in subdir:
            raise RuntimeError("Can only prune based on evaluation or episodic rewards")

        '''
        We do a few things here
        1) remove any trajectories which are < MIN_LENGTH
        2) compute the mean of the last RUN_HEURISTIC steps in each trajectory
        '''
        rank_array = [(np.mean(k[0][-RUN_HEURISTIC:]), k[1], k[0]) for idx, k in enumerate(zip(score_arrays, run_titles)) if k[0].shape[1] > MIN_LENGTH]

        '''
        Sort the trajectories based on the RUN_HEURISTIC score
        '''
        rank_array = sorted(rank_array, key=lambda x: x[0], reverse=True)

        '''
        Get the N_BEST trajectories.
        '''
        rank_array = rank_array[:N_BEST]
        perf_heuristics, good_run_titles, score_arrays = (list(l) for l in zip(*rank_array))

    plt.figure(figsize=(4,3))

    [
        generate_plot(score_array, run_title, smoothen=smoothen)
        for score_array, run_title in zip(score_arrays, good_run_titles)
    ]

    plt.ylabel(subdir.replace("_", " "))
    plt.xlabel("Episode")
    
    #legend = plt.legend()
   
    #legend = plt.legend(loc=(1.04,0))
    #export_legend(legend)

    #plt.show()
    plt.title("Walker2d")

    plt.savefig("./walker.pdf", format="pdf", bbox_inches="tight")


def main():
    """
    Change these options and directories to suit your needs
    """
    ## Defaults
    subdir = "evaluation_rewards"
    smoothen = True
    min_length = -1
    only_longest = False
    cumulative = False
    all_seeds = False

    '''
    Set this flag to true if you are pointing the plot_learning_curve script to an onager experiment folder
    The use case of this is if you have >> (many many) run folders you want to comb through for the best performance.
    '''
    USE_ONAGER = False

    ## Options (Uncomment as needed)
    # smoothen = True
    # min_length = 300
    # only_longest = True
    # cumulative = True
    # all_seeds = True

    experiment_name = "/home/sreehari/Downloads/ICLR/results/Walker"
    #experiment_name = "/home/sreehari/Downloads/Onager Sweeps Rainbow RBFDQN/Ant2/"

    run_titles = get_all_run_titles(experiment_name)
    make_graphs(experiment_name,
                subdir,
                run_titles=run_titles,
                smoothen=smoothen,
                min_length=min_length,
                only_longest=only_longest,
                cumulative=cumulative,
                all_seeds=all_seeds,
                use_onager=USE_ONAGER
                )


if __name__ == '__main__':
    main()