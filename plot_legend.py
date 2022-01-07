import matplotlib.pyplot as plt

def export_legend(labels, exps, savepath):
    """
    Export a legend in a borderless, horizontal format
    """
    plt.rcParams.update({'font.size': 20})
    ## Create a fake axes to use
    x = [0, 1]
    y = [0, 1]
    for exp in exps:
        plt.plot(x,y,linewidth=3, label=labels[exp][0], color=labels[exp][1])
    ax = plt.gca()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False, loc='lower center', ncol=len(exps),)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(savepath, dpi="figure", bbox_inches=bbox)

if __name__ == "__main__":
    ## How to use this plotting legend:
    ## (1) Labels is a dict that has keys as experiment names
    ##     and values as a tuple of ('Label Text', 'Color')
    ## (2) You can extend the tuple if you want to style the lines -- just make sure to make the proper
    ##     adjustments in the plt.plot() in the function above.
    ## (3) It would be nice to figure out how to get a multi-line legend, or specify the legend to
    ##     overflow onto another line
    labels = {'vanilla': ('Vanilla', 'dimgrey'),
              'double': ('Double', 'blueviolet'),
              'per': ('PER', 'dodgerblue'),
              'noisy': ('Noisy', 'firebrick'),
              'distributional': ('Distributional', 'darkorange'),
              'dueling': ('Dueling', 'limegreen'),
              'multi-step': ('N-Step', 'khaki'),
              }
    exps = ["vanilla",
        "noisy",
        "per",
        "double",
        "distributional",
        "dueling",
        "multi-step"]

    export_legend(labels, exps, "legend.pdf")
