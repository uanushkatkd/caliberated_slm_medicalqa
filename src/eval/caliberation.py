# src/eval/calibration.py

import numpy as np
import matplotlib.pyplot as plt


def compute_calibration_bins(confidences, correct, n_bins=10):
    confidences = np.array(confidences)
    correct = np.array(correct)

    bins = np.linspace(0, 1, n_bins + 1)

    bin_acc = []
    bin_conf = []
    bin_counts = []

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])

        if mask.sum() == 0:
            bin_acc.append(0)
            bin_conf.append(0)
            bin_counts.append(0)
            continue

        bin_acc.append(correct[mask].mean())
        bin_conf.append(confidences[mask].mean())
        bin_counts.append(mask.sum())

    return bins, np.array(bin_acc), np.array(bin_conf), np.array(bin_counts)


def plot_reliability_diagram(confidences, correct, save_path):
    bins, bin_acc, bin_conf, _ = compute_calibration_bins(confidences, correct)

    plt.figure()
    plt.plot([0, 1], [0, 1])  # perfect calibration
    plt.plot(bin_conf, bin_acc, marker="o")

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")

    plt.savefig(save_path)
    plt.close()


def plot_confidence_histogram(confidences, save_path):
    plt.figure()
    plt.hist(confidences, bins=10)

    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title("Confidence Distribution")

    plt.savefig(save_path)
    plt.close()