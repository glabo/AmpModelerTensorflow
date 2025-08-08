import argparse
import json

from numpy import mean, median

def print_split_means(name, arr, num_splits):
    num_epochs = len(arr)
    if num_epochs < 100:
        num_splits = num_epochs // 10

    split_len = num_epochs // num_splits
    i = 0
    split_means = []
    print(name, "Splits:")
    for _ in range(num_splits):
        split_mean = mean(arr[i:i+split_len])
        split_means.append(split_mean)
        print("    [", i, ":", i+split_len, f"]: \t{split_mean:.4f}")
        i += split_len+1

    return split_means

def print_split_diffs(name, s1, s2):
    print(name, ":")
    for v1, v2 in zip(s1, s2):
        print(f"    {v1-v2:.4f}")

def print_overall_stat(name, arr):
    print(name, ":")

    minimum = min(arr)
    min_loc = arr.index(minimum)
    maximum = max(arr)
    max_loc = arr.index(maximum)
    avg = mean(arr)
    med = median(arr)
    #med_idx = arr.index(med)
    print(f"    Min: {minimum:.4f} @ {min_loc}   Max: {maximum:.4f} @ {max_loc}")
    print(f"    Average: {avg:.4f}   Median: {med:.4f}")

def stats_eval(args):
    stats_file = open(args.stats_path, "r")

    stats = json.load(stats_file)

    loss = stats['loss']
    val_loss = stats['val_loss']

    print("Overall Stats -----------------------")
    print_overall_stat("Loss", loss)
    print_overall_stat("Validation Loss", val_loss)

    if args.splits:
        print()
        print("Split Stats -----------------------")
        loss_split_means = print_split_means("Testing Loss", loss, 10)
        val_loss_split_means = print_split_means("Validation Loss", val_loss, 10)
        print_split_diffs("loss - val_loss", loss_split_means, val_loss_split_means)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_path")
    parser.add_argument("--splits", action="store_true")
    args = parser.parse_args()
    stats_eval(args)
