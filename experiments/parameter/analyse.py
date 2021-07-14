from argformat import StructuredFormatter
import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=StructuredFormatter)
    parser.add_argument('files', nargs='+', help='input files to analyse')
    args = parser.parse_args()

    data = {}

    # Read all parameter files
    for file in args.files:
        with open(file) as infile:
            data[file] = infile.read()

    result = dict()

    complexity = None
    window     = None

    accuracy  = None
    precision = None
    recall    = None
    f1_score  = None

    for file, content in data.items():
        for line in content.split('\n'):
            # Strip lines
            line = line.strip()

            # Ignore epochs
            if line.startswith('[Epoch'):
                continue

            # Get parameters
            if line.startswith('Complexity'):
                complexity = int(line.split()[2].split(',')[0])
                window     = int(line.split()[-1])

            # Extract metrics
            if line.startswith('accuracy'):
                accuracy = float(line.split()[1])
            if line.startswith('weighted avg'):
                precision, recall, f1_score = [float(x) for x in line.split()[2:-1]]

                result[(complexity, window)] = {
                    'accuracy' : accuracy,
                    'precision': precision,
                    'recall'   : recall,
                    'f1_score' : f1_score,
                }

    table = list()
    complexity = None

    for key, value in sorted(result.items()):
        if key[0] != complexity:
            complexity = key[0]
            print(complexity)
            table.append(list())
        table[-1].append(value.get('f1_score'))

    for row in table:
        print(' & '.join("{:.4f}".format(x) for x in row) + ' \\\\')

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    table = np.asarray(table)

    plt.rcParams["font.family"] = "serif"

    fig, ax = plt.subplots()
    im = ax.imshow(table, cmap='bone', vmin=0.8, vmax=1.0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)

    # Create colorbar
    cbar = ax.figure.colorbar(im, cax=cax, ticks=[0.80, 0.85, 0.90, 0.95, 1.00])
    cbar.ax.set_yticklabels(['0.80', '0.85', '0.90', '0.95', '1.00'])
    cbar.ax.set_ylabel("F1-score", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_xticklabels(np.arange(table.shape[1])+1)
    ax.set_yticklabels(2**np.arange(table.shape[0]))
    ax.set_xlabel("window size")
    ax.set_ylabel("complexity")
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.savefig('parameter_selection.png', dpi=300, bbox_inches='tight')
