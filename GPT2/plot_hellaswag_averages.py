import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# Directory containing the logs
log_dir = "logs"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times new roman"]
plt.rcParams["font.size"] = 17
plt.rcParams.update(
    {
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "grid.color": "lightgray",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "axes.grid": True,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def smooth_data(x, y, window=7):
    """
    Smooth data using Savitzky-Golay filter.
    window should be odd and >= 3
    """
    if len(y) < window:
        return y
    return savgol_filter(y, window, 3)  # window length 7, polynomial order 3


def parse_log(file_path):
    hellaswag_iters, hellaswag_accuracies = [], []
    with open(file_path, "r") as file:
        for line in file:
            if "Hellaswag accuracy" in line:
                parts = line.strip().split("|")
                iter_num = int(parts[1].strip().split()[1])
                accuracy = float(parts[2].strip().split()[2])
                hellaswag_iters.append(iter_num)
                hellaswag_accuracies.append(accuracy)
    return hellaswag_iters, hellaswag_accuracies


def aggregate_hellaswag_data(
    file_prefix, num_runs, max_iteration, smooth=False, window=7
):
    all_accuracies = []
    common_iters = None

    for i in range(1, num_runs + 1):
        file_path = os.path.join(log_dir, f"{file_prefix}_{i}.txt")
        hellaswag_iters, hellaswag_accuracies = parse_log(file_path)

        # Limit iterations to the end of the first epoch
        hellaswag_iters = [it for it in hellaswag_iters if it <= max_iteration]
        hellaswag_accuracies = hellaswag_accuracies[: len(hellaswag_iters)]

        if common_iters is None:
            common_iters = hellaswag_iters

        if smooth:
            hellaswag_accuracies = smooth_data(
                hellaswag_iters, hellaswag_accuracies, window
            )

        all_accuracies.append(hellaswag_accuracies)

    # Convert to numpy array for easier calculation
    all_accuracies = np.array(all_accuracies)

    # Calculate mean and standard deviation across runs
    avg_accuracies = np.mean(all_accuracies, axis=0)
    std_accuracies = np.std(all_accuracies, axis=0)

    return common_iters, avg_accuracies, std_accuracies


# Number of runs and smoothing parameters
num_runs = 5
do_smoothing = False  # Set to False to show raw data
window_size = 7  # Must be odd number, increase for more smoothing

# Aggregate data with standard deviations
fineweb_iters, fineweb_avg, fineweb_std = aggregate_hellaswag_data(
    "f", num_runs, 18795, smooth=do_smoothing, window=window_size
)
exquisiteweb50_iters, exquisiteweb50_avg, exquisiteweb50_std = aggregate_hellaswag_data(
    "e_50", num_runs, 18795, smooth=do_smoothing, window=window_size
)
exquisiteweb70_iters, exquisiteweb70_avg, exquisiteweb70_std = aggregate_hellaswag_data(
    "e_90", num_runs, 18795, smooth=do_smoothing, window=window_size
)

"""
fineweb_iters, fineweb_avg, fineweb_std = aggregate_hellaswag_data(
    "f", num_runs, 18795, smooth=do_smoothing, window=window_size
)
exquisiteweb50_iters, exquisiteweb50_avg, exquisiteweb50_std = aggregate_hellaswag_data(
    "e_50", num_runs, 17209, smooth=do_smoothing, window=window_size
)
exquisiteweb70_iters, exquisiteweb70_avg, exquisiteweb70_std = aggregate_hellaswag_data(
    "e_90", num_runs, 14060, smooth=do_smoothing, window=window_size
)
"""

# GPT-2 Checkpoint accuracy
gpt2_checkpoint_accuracy = 0.294463


# Find crossing points
def find_crossing_point(iters, accuracies, threshold):
    for idx, acc in enumerate(accuracies):
        if acc > threshold:
            return idx, iters[idx]
    return None, None


print("\nGPT-2 Checkpoint Crossing Points:")
print(
    f"FineWeb crosses at iter {find_crossing_point(fineweb_iters, fineweb_avg, gpt2_checkpoint_accuracy)[1]}"
)
print(
    f"ExquisiteWeb-50 crosses at iter {find_crossing_point(exquisiteweb50_iters, exquisiteweb50_avg, gpt2_checkpoint_accuracy)[1]}"
)
print(
    f"ExquisiteWeb-70 crosses at iter {find_crossing_point(exquisiteweb70_iters, exquisiteweb70_avg, gpt2_checkpoint_accuracy)[1]}"
)

# Plotting
plt.figure(figsize=(8, 6))

# Plot FineWeb with confidence band
plt.plot(
    fineweb_iters,
    fineweb_avg,
    color="cornflowerblue",
    label="FineWeb-10BT",
    linewidth=2.5,
)
plt.fill_between(
    fineweb_iters,
    fineweb_avg - fineweb_std,
    fineweb_avg + fineweb_std,
    color="cornflowerblue",
    alpha=0.2,
)
plt.scatter(
    [fineweb_iters[-1]],
    [fineweb_avg[-1]],
    color="cornflowerblue",
    marker="o",
    s=80,
    label="_nolegend_",
    zorder=10,
    edgecolors="black",
    linewidth=1.5,
)

# Plot ExquisiteWeb-50 with confidence band

plt.plot(
    exquisiteweb50_iters,
    exquisiteweb50_avg,
    color="green",
    label="FinerWeb-10BT (-8%, 0.5 quality threshold)",
    linewidth=2.5,
)
plt.fill_between(
    exquisiteweb50_iters,
    exquisiteweb50_avg - exquisiteweb50_std,
    exquisiteweb50_avg + exquisiteweb50_std,
    color="green",
    alpha=0.2,
)
plt.scatter(
    [17209],
    [exquisiteweb70_avg[62]],
    color="green",
    marker="o",
    s=80,
    label="_nolegend_",
    zorder=10,
    edgecolors="black",
    linewidth=1.5,
)

# Plot ExquisiteWeb-70 with confidence band
plt.plot(
    exquisiteweb70_iters,
    exquisiteweb70_avg,
    color="orange",
    label="FinerWeb-10BT (-25%, 0.9 quality threshold)",
    linewidth=2.5,
)
plt.fill_between(
    exquisiteweb70_iters,
    exquisiteweb70_avg - exquisiteweb70_std,
    exquisiteweb70_avg + exquisiteweb70_std,
    color="orange",
    alpha=0.2,
)
plt.scatter(
    [14060],
    [exquisiteweb70_avg[60]],
    color="orange",
    marker="o",
    s=80,
    label="_nolegend_",
    zorder=10,
    edgecolors="black",
    linewidth=1.5,
)

# GPT-2 Checkpoint
plt.axhline(
    y=gpt2_checkpoint_accuracy,
    color="black",
    linestyle="--",
    label="OpenAI GPT-2 checkpoint",
)

plt.gca().xaxis.set_major_locator(
    plt.MultipleLocator(2000)
)  # X-axis major ticks every 2000 iterations
plt.gca().yaxis.set_major_locator(
    plt.MultipleLocator(0.01)
)  # Y-axis major ticks every 0.01


plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("hellaswag.pdf", bbox_inches="tight")
