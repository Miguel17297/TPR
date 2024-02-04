import numpy as np
import argparse
from sklearn.decomposition import PCA
from models.models import OneClassSVM, IsolationForest, StatisticalModel
import warnings
from models.utils import dataset_division
import os
import sys

warnings.filterwarnings('ignore')


def compute(features_linked, features_trip, features_tb, features_bot, results_path, outfile, pca_value=None):

    # dataset division
    train_linked, test_linked = dataset_division(features_linked)
    train_trip, test_trip = dataset_division(features_trip)
    train_taobao, test_taobao = dataset_division(features_tb)
    train_bot, test_bot = dataset_division(features_bot, 0.5)

    train_normal = np.vstack((train_linked, train_trip, train_taobao))
    test_normal = np.vstack((test_linked, test_trip, test_taobao))

    labels_normal_test = np.ones((len(test_normal), 1)) * 1
    labels_bot_test = np.ones((len(test_bot), 1)) * -1

    test_data = np.vstack((test_normal, test_bot))
    test_labels = np.vstack((labels_normal_test, labels_bot_test))

    if pca_value:
        pca = PCA(n_components=pca_value)
        train_normal = pca.fit_transform(train_normal)
        test_data = pca.transform(test_data)


    original_stdout = sys.stdout

    with open(outfile, "w") as f:
        sys.stdout = f

        # Models

        print('\n-- Anomaly Detection based on Statistical Model --')

        anomaly_threshold = [5, 10, 3, 7]

        sm = StatisticalModel()
        sm.hyper_tunning(train_normal, test_data, test_labels, anomaly_threshold)

        print('-----------------------------------------------------------------\n')

        print('\n-- Anomaly Detection based on One Class Support Vector Machines--')

        nu = [0.1, 0.4, 0.5, 0.8]

        ocsvm = OneClassSVM("linear")
        ocsvm.hyper_tunning(train_normal, test_data, test_labels, nu)

        rbf_ocsvm = OneClassSVM("rbf")
        rbf_ocsvm.hyper_tunning(train_normal, test_data, test_labels, nu)

        poly_ocsvm = OneClassSVM("poly")
        poly_ocsvm.hyper_tunning(train_normal, test_data, test_labels, nu)

        print('-----------------------------------------------------------------\n')

        print('\n-- Anomaly Detection based on Isolation Forest--')

        max_samples = [100, 200, 300, 400]
        random_state = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        isf = IsolationForest()
        isf.hyper_tunning(train_normal, test_data, test_labels, max_samples, random_state)

    sys.stdout = original_stdout


def main(bot, pca):
    features_bot = np.loadtxt(f'data/bot{bot}.dat')
    features_linked = np.loadtxt("data/linkedIn.dat")
    features_trip = np.loadtxt("data/tripadvisor.dat")
    features_tb = np.loadtxt("data/taobao.dat")

    results_path = os.path.join(os.path.join(os.getcwd()), "results",  f"bot{bot}")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if pca:

        pca_values = [*range(6,28)]

        for i in pca_values:
            file_name = os.path.join(results_path, f'pca_{i}.txt')

            compute(features_linked, features_tb, features_trip, features_bot, results_path, file_name, i)

    else:
        file_name = os.path.join(results_path, "modelValidation.txt")

        compute(features_linked, features_tb, features_trip, features_bot, results_path, file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('-b', '--bot', type=int, help='Bot Level', choices=[1, 2, 3], default=3)
    parser.add_argument('--pca',action="store_true", help='Use pca')
    args = parser.parse_args()

    bot = args.bot
    pca = args.pca
    main(bot, pca)

