import argparse
import json
import matplotlib.pyplot as plt
import os

def plot_stats(stats_file):
    stats_dir = os.path.dirname(stats_file)
    f = open(stats_file)
    stats = json.load(f)
        
    loss = stats['loss']
    val_loss = stats['val_loss']
    num_epochs = len(loss)
    epochs = [e for e in range(num_epochs)]

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(epochs, loss, label="Loss")
    ax1.plot(epochs, val_loss, label="Validation Loss")
    ax1.legend()
    ax1.grid("on")

    val_loss_mins = [x for i, x in enumerate(val_loss) if x == min(val_loss[:i+1])]
    mins = [m for m in range(len(val_loss_mins))]
    ax2.plot(mins, val_loss_mins, label="Validation Loss Gradient")
    ax2.legend()
    ax2.grid("on")

    plt.savefig(stats_dir + "/loss_plot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_file")
    args = parser.parse_args()
    plot_stats(args.stats_file)
