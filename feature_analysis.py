from sklearn.decomposition import PCA
import argparse
import numpy as np
import os
import sys
import itertools
import matplotlib.pyplot as plt

def feature_analysis(bot):

    features_bot = np.loadtxt(f'data/bot{bot}.dat')
    features_linked = np.loadtxt("data/linkedIn.dat")
    features_trip = np.loadtxt("data/tripadvisor.dat")
    features_tb = np.loadtxt("data/taobao.dat")

    results_path = os.path.join(os.path.join(os.getcwd()), "results", f"bot{bot}")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # data analysis
    features_normal = np.vstack((features_linked, features_trip, features_tb))
    features = np.vstack((features_normal, features_bot))
    labels_normal = np.ones((len(features_normal), 1)) * 1
    labels_bot = np.ones((len(features_bot), 1)) * -1

    labels = np.vstack((labels_normal, labels_bot))

    obs, n_fea = features.shape

    '''i need to make all comvbination of features and plot them'''
    comb = [*(itertools.combinations(range(n_fea), 2))]

    f_labels = np.loadtxt("data/feature_labels.csv",
                          delimiter=",", dtype=str)

    # Create a grid of subplots based on the number of combinations
    num_cols = 7  # You can adjust the number of columns as needed
    num_rows = 6

    # Set the number of subplots per file
    subplots_per_file = num_rows * num_cols  # You can adjust this based on your preference

    # Create a list of combinations for each file
    # Iterate over combinations and create separate files
    for file_idx, combinations in enumerate(
            [comb[i:i + subplots_per_file] for i in range(0, len(comb), subplots_per_file)]):
        # Create a new figure with subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 10))  # Adjust the width and height as needed

        # Plot each combination in a subplot
        # Plot each combination in a subplot
        for idx, (i, j) in enumerate(combinations):
            row = idx // num_cols
            col = idx % num_cols
            axes[row, col].scatter(features[:, i], features[:, j], c=labels.flatten(), cmap='viridis')
            axes[row, col].set_title(f'{f_labels[i]} vs {f_labels[j]}')

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        f_name = os.path.join(results_path,
                              f'combined_feature_plots_{file_idx + 1}.png')

        # Save the current figure
        plt.savefig(f_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('-b', '--bot', type=int, help='Bot Level', choices=[1, 2, 3], default=3)
    args = parser.parse_args()

    bot = args.bot
    feature_analysis(bot)