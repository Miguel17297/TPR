import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models import OneClassSVM, IsolationForest
import warnings
from utils import validate_model, distance, waitforEnter, plotFeatures, dataset_division

warnings.filterwarnings('ignore')

# TODO: Testar o codigo

def compute(features_normal, features_bot):
    # TODO: Colocar aqui o codigo dos graficos para cada um dos pares de features

    """
        Obs, nFea = features_bot.shape

        plt.figure(1)
        plotFeatures(features, oClass, 0,nFea-1)

    """

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
