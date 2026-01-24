#For plots_1d, inserted for Truth H pT plot before savefig

        ax.text(
            ax.get_xlim()[1]*0.7,                    # x position (slightly to the right of the line)
            ax.get_ylim()[1]*0.5,   # y position (80% of y-axis top)
            f"≥{300} GeV: {frac}%",  # text
            color="k",
            fontsize=18,
            rotation=0,
        )
    ax.axvline(
    x=300,             # position
    color='k',       # choose color
    linestyle='--',    # dashed line
    linewidth=1.5,     # optional, thickness
    label=None         # optional: you can add a label for the line in the legend
    )

edges = h.axes[0].edges
counts = h.values()
mask = edges[:-1] >= 300 
frac = round(counts[mask].sum()*100, 2)