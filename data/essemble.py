import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models import OneClassSVM, IsolationForest
import itertools
import warnings
from utils import distance, dataset_division,validate_model

import os
import sys

def ensemble(bot, pca, outfile):
    print(outfile)
    original_stdout = sys.stdout
    
    results_path = os.path.join(os.path.dirname(os.getcwd()), "results")
    

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
            
    with open(os.path.join(results_path,"ensemble.txt"), "w") as f:
        sys.stdout = f
        
    
        features_bot = np.loadtxt(f'bot{bot}.dat')
        features_normal = np.loadtxt("features.out")
        train_normal, test_normal = dataset_division(features_normal)
        train_bot, test_bot = dataset_division(features_bot)
        labels_normal_test = np.ones((len(test_normal), 1)) * 1
        labels_bot_test = np.ones((len(test_bot), 1)) * -1

        test_data = np.vstack((test_normal, test_bot))
        test_labels = np.vstack((labels_normal_test, labels_bot_test))
        

        print('-----------------------------------------------------------------\n')

        print('\n-- Anomaly Detection based on One Class Support Vector Machines--')

        nu = 0.1

        ocsvm = OneClassSVM("linear",nu)
        ocsvm.train(train_normal)

        rbf_ocsvm = OneClassSVM("rbf",nu)
        # rbf_ocsvm.hyper_tunning(train_normal, test_data, test_labels, nu)
        rbf_ocsvm.train(train_normal)

        poly_ocsvm = OneClassSVM("poly",nu)
        # poly_ocsvm.hyper_tunning(train_normal, test_data, test_labels, nu)
        rbf_ocsvm.train(train_normal)

        print('-----------------------------------------------------------------\n')
        
        print('\n-- Anomaly Detection based on Isolation Forest--')

        max_samples =  400
        random_state = 9

        isf = IsolationForest(max_samples, random_state)
        # isf.hyper_tunning(train_normal, test_data, test_labels, max_samples, random_state)
        isf.train(train_normal)
        
        results = np.ones(len(test_data))
        
        for i in test_data:
            if ocsvm.predict(i) == 1 and rbf_ocsvm.predict(i) == 1 and isf.predict(i) == 1:
                results[i] = 1
            elif centroids(train_normal,i) == 1 and isf.predict(i) == 1:
                results[i] = -1
                
        validate_model(results.reshape(-1, 1), test_labels)
        
    sys.stdout = original_stdout
    
def centroids(train_normal,test_data):
    print('\n-- Anomaly Detection based on Statistical Model --')

    percentage = 0.7

    centroids = np.mean(train_normal, axis=0)
    print('All Features Centroids:\n', centroids)

    anomaly_threshold = 7
    print('\n-- Anomaly Detection based on Centroids Distances --')
    dists = [distance(x, centroids)]
    return 1 if min(dists) > anomaly_threshold else -1


    
    
    

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
