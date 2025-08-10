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

    plt.plot(epochs, loss, label="Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.legend()
    plt.grid("on")
    plt.savefig(stats_dir + "/loss_plot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_file")
    args = parser.parse_args()
    plot_stats(args.stats_file)
