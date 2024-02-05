import numpy as np
import argparse
from sklearn.decomposition import PCA
from models import OneClassSVM, IsolationForest, StatisticalModel
from models.utils import dataset_division,validate_model

import os
import sys

BEST_NU_1 = 0.1
BEST_NU_2 = 0.4
BEST_NU_3 = 0.8
BEST_THRESHOLD = 3
BEST_MS = 400
BEST_RS = 0

def ensemble(bot):
    original_stdout = sys.stdout
    
    results_path = os.path.join(os.path.join(os.getcwd()), "results", "essemble")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    features_linked = np.loadtxt("data/linkedIn.dat")
    features_trip = np.loadtxt("data/tripadvisor.dat")
    features_tb = np.loadtxt("data/taobao.dat")
    
    train_linked, test_linked = dataset_division(features_linked)
    train_trip, test_trip = dataset_division(features_trip)
    train_taobao, test_taobao = dataset_division(features_tb)
    
    features_bot = np.loadtxt(f'data/bot{bot}.dat')

    train_bot, test_bot = dataset_division(features_bot, 0.5)
    train_normal = np.vstack((train_linked, train_trip, train_taobao))

    test_normal = np.vstack((test_linked, test_trip, test_taobao))
    

    labels_normal_test = np.ones((len(test_normal), 1)) * 1
    labels_bot_test = np.ones((len(test_bot), 1)) * -1

    test_data = np.vstack((test_normal, test_bot))
    test_labels = np.vstack((labels_normal_test, labels_bot_test))
    
    pca_27= PCA(n_components=27)
    pca_19 = PCA(n_components=19)
    pca_16 = PCA(n_components=16)
    pca_26 = PCA(n_components=26)

    train_normal_pca_27= pca_27.fit_transform(train_normal)
    test_normal_pca_27 = pca_27.transform(test_data)
    

    train_normal_pca_19= pca_19.fit_transform(train_normal)
    test_normal_pca_19 = pca_19.transform(test_data)
    
    train_normal_pca_16= pca_16.fit_transform(train_normal)
    test_normal_pca_16 = pca_16.transform(test_data)
    
    train_normal_pca_26= pca_26.fit_transform(train_normal)
    test_normal_pca_26 = pca_26.transform(test_data)

    sm = StatisticalModel(BEST_THRESHOLD)
    sm.train(train_normal)
    sm_res = sm.predict(test_data)

    ocsvm = OneClassSVM("linear", BEST_NU_2)
    ocsvm.train(train_normal_pca_19)
    ocsvm_res = ocsvm.predict(test_normal_pca_19)

    rbf_ocsvm = OneClassSVM("rbf", BEST_NU_1)
    rbf_ocsvm.train(train_normal_pca_16)
    rbf_res = rbf_ocsvm.predict(test_normal_pca_16)

    poly_ocsvm = OneClassSVM("poly", BEST_NU_3)
    poly_ocsvm.train(train_normal_pca_26)
    poly_res = poly_ocsvm.predict(test_normal_pca_26)

    isf = IsolationForest(BEST_MS, BEST_RS)
    isf.train(train_normal_pca_27)
    isf_res = isf.predict(test_normal_pca_27)

    essemble_res = np.ones(len(test_data))
    

    with open(os.path.join(results_path,"ensemble.txt"), "w") as f:
        sys.stdout = f

        print('-----------------------------------------------------------------\n')

        print('\n-- Anomaly Detection based on Essemble--')

        n_obs, _ = test_data.shape

        for i in range(n_obs):
            if ocsvm_res[i] == -1 and poly_res[i] == -1 and isf_res[i] == -1:
                essemble_res[i] = -1
            elif rbf_res[i] == 1 and sm_res[i] == 1:
                essemble_res[i] = 1
                
        validate_model(essemble_res.reshape(-1, 1), test_labels)
        
    sys.stdout = original_stdout
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('-b', '--bot', type=int, help='Bot Level', choices=[1, 2, 3], default=3)
    args = parser.parse_args()
    
    bot = args.bot
    ensemble(bot)
