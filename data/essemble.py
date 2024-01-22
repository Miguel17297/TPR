import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models import OneClassSVM, IsolationForest, StatisticalModel
import itertools
import warnings
from utils import distance, dataset_division,validate_model

import os
import sys

BEST_NU = 0.5
BEST_THRESHOLD = 7
BEST_MS = 400
BEST_RS = 9

def ensemble(bot, pca, outfile):
    original_stdout = sys.stdout
    
    results_path = os.path.join(os.path.dirname(os.getcwd()), "results")
    

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    features_bot = np.loadtxt(f'bot{bot}.dat')
    features_normal = np.loadtxt("features.out")
    train_normal, test_normal = dataset_division(features_normal)
    train_bot, test_bot = dataset_division(features_bot)
    labels_normal_test = np.ones((len(test_normal), 1)) * 1
    labels_bot_test = np.ones((len(test_bot), 1)) * -1

    test_data = np.vstack((test_normal, test_bot))
    test_labels = np.vstack((labels_normal_test, labels_bot_test))

    sm = StatisticalModel(BEST_THRESHOLD)
    sm.train(train_normal)
    sm_res = sm.predict(test_data)

    ocsvm = OneClassSVM("linear", BEST_NU)
    ocsvm.train(train_normal)
    ocsvm_res = ocsvm.predict(test_data)

    rbf_ocsvm = OneClassSVM("rbf", BEST_NU)
    rbf_ocsvm.train(train_normal)
    rbf_res = rbf_ocsvm.predict(test_data)

    poly_ocsvm = OneClassSVM("poly", BEST_NU)
    poly_ocsvm.train(train_normal)
    poly_res = poly_ocsvm.predict(test_data)

    isf = IsolationForest(BEST_MS, BEST_RS)
    isf.train(train_normal)
    isf_res = isf.predict(test_data)

    essemble_res = np.ones(len(test_data))

    with open(os.path.join(results_path,"ensemble.txt"), "w") as f:
        sys.stdout = f


        print('-----------------------------------------------------------------\n')

        print('\n-- Anomaly Detection based on Essemble--')

        n_obs, _ = test_data.shape

        for i in range(n_obs):
            if ocsvm_res[i] == 1 and rbf_res[i] == 1 and poly_res[i] == 1:
                essemble_res[i] = 1
            elif  sm_res[i] == -1 and isf_res[i] == -1:
                essemble_res[i] = -1
                
        validate_model(essemble_res.reshape(-1, 1), test_labels)
        
    sys.stdout = original_stdout
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('-b', '--bot', type=int, help='Bot Level', choices=[1, 2, 3], default=3)
    parser.add_argument('-o', '--output', type=str,nargs='?', help='Output file',const='ensemble.txt')
    parser.add_argument('--pca', type=bool, help='Use pca', default=False)
    args = parser.parse_args()
    

    bot = args.bot
    outfile = args.output
    pca = args.pca
    
    ensemble(bot, pca, outfile)
