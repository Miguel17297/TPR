import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models import OneClassSVM, IsolationForest
import itertools
import warnings
from utils import distance, dataset_division
import os

warnings.filterwarnings('ignore')


def compute(features_normal, features_bot):

    results_path = os.path.join(os.path.dirname(os.getcwd()), "results")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # data analysis

    features = np.vstack((features_normal, features_bot))
    labels_normal = np.ones((len(features_normal), 1)) * 1
    labels_bot= np.ones((len(features_bot), 1)) * -1

    labels = np.vstack((labels_normal, labels_bot))

    obs, n_fea = features.shape

    '''i need to make all comvbination of features and plot them'''
    comb = [*(itertools.combinations(range(n_fea), 2))]

    f_labels = np.loadtxt("feature_labels.csv",
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


    # dataset division

    train_normal, test_normal = dataset_division(features_normal)
    train_bot, test_bot = dataset_division(features_bot)
    labels_normal_test = np.ones((len(test_normal), 1)) * 1
    labels_bot_test = np.ones((len(test_bot), 1)) * -1

    test_data = np.vstack((test_normal, test_bot))
    test_labels = np.vstack((labels_normal_test, labels_bot_test))

    # Models

    print('\n-- Anomaly Detection based on Statistical Model --')

    percentage = 0.7

    centroids = np.mean(train_normal, axis=0)
    print('All Features Centroids:\n', centroids)

    AnomalyThreshold = [5, 10, 3, 7]

    for j in AnomalyThreshold:
        print('\n-- Anomaly Detection based on Centroids Distances --')
        nObsTest, nFea = test_data.shape
        results = np.ones(nObsTest)
        for i in range(nObsTest):
            x = test_data[i]
            dists = [distance(x, centroids)]
            if min(dists) > j:
                results[i] = -1
            else:
                results[i] = 1

    # TODO: Codigo de validar o modelo estatistico
    """
        Codigo de validar o modelo estatistico

    """

    print('\n-- Anomaly Detection based on One Class Support Vector Machines--')

    nu = [0.1, 0.4, 0.5, 0.8]

    ocsvm = OneClassSVM("linear")
    ocsvm.hyper_tunning(train_normal, test_data, test_labels, nu)

    rbf_ocsvm = OneClassSVM("rbf")
    rbf_ocsvm.hyper_tunning(train_normal, test_data, test_labels, nu)

    poly_ocsvm = OneClassSVM("poly")
    poly_ocsvm.hyper_tunning(train_normal, test_data, test_labels, nu)

    print('\n-- Anomaly Detection based on Isolation Forest--')

    max_samples = [100, 200, 300, 400]
    random_state = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    isf = IsolationForest()
    isf.hyper_tunning(train_normal, test_data, test_labels, max_samples, random_state)


def main(bot, pca):
    features_bot = np.loadtxt(f'bot{bot}.dat')
    features_normal = np.loadtxt("features.out")

    if pca:
        pca_values = ([2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        for i in pca_values:
            pca = PCA(n_components=i)
            pca_normal = pca.fit_transform(features_normal)
            pca_bot = pca.fit_transform(features_bot)
            
            compute(pca_normal, pca_bot)

    else:

        compute(features_normal, features_bot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('-b', '--bot', type=int, help='Bot Level', choices=[1, 2, 3], default=3)
    parser.add_argument('--pca', type=bool, help='Use pca', default=False)
    args = parser.parse_args()

    bot = args.bot
    pca = args.pca

    main(bot, pca)
