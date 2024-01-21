import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from scipy.stats import multivariate_normal
from sklearn import svm
import itertools
import time
import sys
import warnings

warnings.filterwarnings('ignore')


def waitforEnter(fstop=True):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")


## -- 3 -- ##
def plotFeatures(features, oClass, f1index=0, f2index=1):
    nObs, nFea = features.shape
    colors = ['b', 'r']
    for i in range(nObs):
        plt.plot(features[i, f1index], features[i, f2index], 'o' + colors[int(oClass[i])])

    plt.show()
    waitforEnter()


def logplotFeatures(features, oClass, f1index=0, f2index=1):
    nObs, nFea = features.shape
    colors = ['b', 'g', 'r']
    for i in range(nObs):
        plt.loglog(features[i, f1index], features[i, f2index], 'o' + colors[int(oClass[i])])

    plt.show()
    waitforEnter()


def validate_model(model_predictions, real_values):
    # TODO: arranjar classes de dados (real_values -> normal e anamolo) de forma a ser uma lista de 1 (normal) e -1 (anomalo)
    """
    Validates model performance
    :param model_predictions: predictions (anomaly, normal) made by a model for a given data
    :param real_values: true data classification (equivale Classes[o3testClass[i][0]])
    """

    assert len(model_predictions) == len(real_values)
    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(len(model_predictions)):

        if model_predictions[i][0] == 1 and real_values[i][0] == 1:  # true positive
            tp = tp + 1
        elif model_predictions[i][0] == 1 and real_values[i][0] == 0:  # false positives
            fp = fp + 1
        elif model_predictions[i][0] == -1 and real_values[i][0] == 0:  # true negative
            tn = tn + 1

        else:
            fn = fn + 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    if recall + precision == 0:
        f1_score = 0
    else:
        f1_score = 2 * (recall * precision) / (recall + precision)

    print(f"True Positives: {tp}, False Positives: {fp}, True Negatives: {tn}, False Negatives: {fp} \n")
    print(f"Acurracy: {accuracy} \n")
    print(f"Recall: {recall} \n")
    print(f"Precision: {precision} \n")
    print(f"F1-Score: {f1_score} \n")

    labels = ["Anomaly", "Normal"]
    results = np.array([tp, fp, tn, fn])

    seaborn.set(font_scale=1.4)
    # TODO : fix heat map
    # ax = seaborn.heatmap(results, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})

    # ax.set_xticklabels(labels)
    # ax.set_yticklabels(labels)

    # ax.set(ylabel="Real Values", xlabel="Predicted Values")
    # plt.show()
    return f1_score


## -- 11 -- ##
def distance(c, p):
    s = 0
    n = 0
    for i in range(len(c)):
        if c[i] > 0:
            s += np.square((p[i] - c[i]) / c[i])
            n += 1

    return (np.sqrt(s / n))

    # return(np.sqrt(np.sum(np.square((p-c)/c))))


########### Main Code #############
Classes = {0: 'Normal', 1: 'Anomaly'}
plt.ion()
nfig = 1

## -- 2 -- ##
# features_bot1 = np.loadtxt("bot1.dat")
# features_bot2 = np.loadtxt("bot2.dat")
features_bot3 = np.loadtxt("bot3.dat")
features_normal = np.loadtxt("features.out")

# oClass_bot1 = np.ones((len(features_bot1), 1)) * 1

oClass_bot3 = np.ones((len(features_bot3), 1)) * 1
print(features_bot3.shape)
# oClass_bot1 = np.ones((len(features_bot3), 1)) * 0

oClass_normal = np.ones((len(features_normal), 1)) * 0
print(features_normal.shape)

oClass_test = np.vstack((oClass_bot3, oClass_normal))

features = np.vstack((features_normal, features_bot3))

oClass = np.vstack((oClass_normal, oClass_bot3))

Obs, nFea = features_bot3.shape
print('Train Silence Features Size:', features.shape)

plt.figure(1)
plotFeatures(features, oClass, 0, nFea - 1)

# sys.exit(0)


## -- 3 -- ##
#:i
percentage = 0.7
# pB = int(len(features_bot1) * percentage)
# trainfeatures_bot1 = features_bot1[:pB, :]
# pYT = int(len(features_bot2) * percentage)
# trainfeatures_bot2 = features_bot2[:pYT, :]
pb3 = int(len(features_bot3) * percentage)
trainfeatures_bot3 = features_bot3[:pb3:]
Pn = int(len(features_normal) * percentage)
trainfeatures_normal = features_normal[: Pn:]
test_features_normal = features_normal[Pn:, :]
test_features_bot3 = features_bot3[pb3:, :]

# i2train = np.vstack((trainfeatures_bot1, trainfeatures_bot2))
# o2trainClass = np.vstack((oClass_browsing[:pB], oClass_yt[:pYT]))
oNormal_trainClass = oClass_normal[: Pn:]
oClass_bot3_test = np.ones((len(test_features_bot3), 1)) * 1
oClass_normal_test = np.ones((len(test_features_normal), 1)) * 0
oClass_test = np.vstack((oClass_bot3_test, oClass_normal_test))

#:ii
# inoCtrain = np.trainfeatures_normal
# o3trainClass = np.vstack((oClass_normal[:Pn]))

#:iii
# testfeatures_bot1 = features_bot1[pB:, :]
# testfeatures_bot2 = features_bot2[pYT:, :]
# testfeatures_bot3 = features_bot3[pb3:, :]

# i3Atest = np.vstack((testfeatures_bot1, testfeatures_bot2, testfeatures_bot3))
# o3testClass = np.vstack((oClass_browsing[pB:], oClass_yt[pYT:], oClass_mining[pM:]))

## -- 7 -- ##

# i2train = np.vstack((trainfeatures_bot1, trainfeatures_bot2))
# scaler = MaxAbsScaler().fit(i2train)
# i2train=scaler.transform(i2train)

centroids = 0

pClass = oNormal_trainClass.flatten()
print(pClass)
centroids = np.mean(trainfeatures_normal, axis=0)
print('All Features Centroids:\n', centroids)

i3Atest = np.vstack((test_features_bot3, test_features_normal))
# i3Atest= scaler.transform(trainfeatures_bot3)

AnomalyThreshold = [5, 10, 3, 7]

o3testClass = np.vstack((oClass_bot3, oClass_normal))
for j in AnomalyThreshold:
    print('\n-- Anomaly Detection based on Centroids Distances --')
    nObsTest, nFea = i3Atest.shape
    results = np.ones(nObsTest)
    for i in range(nObsTest):
        x = i3Atest[i]
        dists = [distance(x, centroids)]
        if min(dists) > j:
            results[i] = -1
        else:
            results[i] = 1

        print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f}] -> Result -> {}'.format(i, Classes[
            o3testClass[i][0]], *dists, results[i]))
        # print(i)
        # print(Classes[o3testClass[i][0]])
        # print(*dists)
        # print(results[i])

## -- 8 -- ##
print('\n-- Anomaly Detection based on One Class Support Vector Machines--')
i2train = trainfeatures_normal
i3Atest = np.vstack((test_features_bot3, test_features_normal))

nu = [0.1, 0.4, 0.5, 0.8]
model_predictions = np.ones(len(i3Atest))
best__ocsvm = -1
best_rbf_ocsvm = -1
best_poly_ocsvm = -1

for j in nu:
    ocsvm = svm.OneClassSVM(gamma='scale', kernel='linear', nu=j).fit(i2train)
    rbf_ocsvm = svm.OneClassSVM(gamma='scale', kernel='rbf', nu=j).fit(i2train)
    poly_ocsvm = svm.OneClassSVM(gamma='scale', kernel='poly', nu=j, degree=2).fit(i2train)

    L1 = ocsvm.predict(i3Atest)
    L2 = rbf_ocsvm.predict(i3Atest)
    L3 = poly_ocsvm.predict(i3Atest)

    AnomResults = {-1: "Anomaly", 1: "OK"}

    # if model_predictions == -1:

    print(o3testClass.shape)
    L_arr = L1.reshape(-1, 1)
    print(L_arr.shape)
    vm_ocsvm = validate_model(L_arr, oClass_test)
    vm_rbf_ocsvm = validate_model(L2.reshape(-1, 1), oClass_test)
    vm_poly_ocsvm = validate_model(L3.reshape(-1, 1), oClass_test)

    validate_model(L_arr, oClass_test)
    if vm_ocsvm > best__ocsvm:
        best__ocsvm = vm_ocsvm
        best_comb = j
        print("OCSVM", best__ocsvm)

    if vm_rbf_ocsvm > best_rbf_ocsvm:
        best_rbf_ocsvm = vm_rbf_ocsvm
        best_comb = j
        print("rbf", best_rbf_ocsvm)

    if vm_poly_ocsvm > best_poly_ocsvm:
        best_poly_ocsvm = vm_poly_ocsvm
        best_comb = j
        print("poly", best_poly_ocsvm)

    nObsTest, nFea = i3Atest.shape
    # for i in range(nObsTest):
    #     # print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i, Classes[
    #     #     o3testClass[i][0]], AnomResults[L1[i]], AnomResults[L2[i]], AnomResults[L3[i]]))

## -- 9 -- ##

# ### Isolation Forest ###
# clf = IsolationForest(max_samples=100, random_state=0)
# clf.fit(i2train)
# x= clf.predict(i3Atest)
# print(x)

# max_samples = [100,200,300,400]
# random_state = [0,1,2,3,4,5,6,7,8,9]
# best = -1
# comb = list(itertools.product(max_samples, random_state))
# for a,b in comb:
#     clf = IsolationForest(max_samples=a, random_state=b)
#     clf.fit(i2train)
#     x= clf.predict(i3Atest)
#     print(x)
#     print(a,b)
#     result = validate_model(x.reshape(-1,1), oClass_test)
#     print(validate_model(x.reshape(-1,1), oClass_test))
#     if result > best:
#         best = result
#         best_comb = (a,b)
#         print(best)


### PCA ###


best_pca = -1
pca_values = ([2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
for i in pca_values:
    pca = PCA(n_components=i)
    pca_model = pca.fit(i2train)
    print("PCA", pca.explained_variance_ratio_)
    pca_transform = pca_model.transform(i2train)
    pca_test = pca_model.transform(i3Atest)
    features_bot3_pca = pca_model.transform(features_bot3)
    features_normal_pca = pca_model.transform(features_normal)
    ### Train Data ###
    pb3 = int(len(features_bot3_pca) * percentage)
    trainfeatures_bot3_pca = features_bot3_pca[:pb3:]
    Pn = int(len(features_normal_pca) * percentage)
    trainfeatures_normal_pca = features_normal_pca[: Pn:]
    test_features_normal_pca = features_normal_pca[Pn:, :]
    test_features_bot3_pca = features_bot3_pca[pb3:, :]

    # i2train = np.vstack((trainfeatures_bot1, trainfeatures_bot2))
    # o2trainClass = np.vstack((oClass_browsing[:pB], oClass_yt[:pYT]))
    oNormal_trainClass_pca = oClass_normal[: Pn:]
    oClass_bot3_test_pca = np.ones((len(test_features_bot3_pca), 1)) * 1
    oClass_normal_test_pca = np.ones((len(test_features_normal_pca), 1)) * 0
    oClass_test_pca = np.vstack((oClass_bot3_test, oClass_normal_test_pca))
    i3Atest_pca = np.vstack((test_features_bot3_pca, test_features_normal_pca))
    i2train_pca = np.vstack((trainfeatures_normal_pca, trainfeatures_bot3_pca))

## Comparing all models transformed by PCA##
### Statistical MODEL ###
for j in AnomalyThreshold:
    print('\n-- Anomaly Detection based on Centroids Distances --')
    nObsTest, nFea = i3Atest_pca.shape
    results = np.ones(nObsTest)
    for i in range(nObsTest):
        x = i3Atest_pca[i]
        dists = [distance(x, centroids)]
        if min(dists) > j:
            results[i] = -1
        else:
            results[i] = 1

        print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f}] -> Result -> {}'.format(i, Classes[
            o3testClass[i][0]], *dists, results[i]))

### ONE CLASS SVM ###
for j in nu:
    ocsvm = svm.OneClassSVM(gamma='scale', kernel='linear', nu=j).fit(i2train_pca)
    rbf_ocsvm = svm.OneClassSVM(gamma='scale', kernel='rbf', nu=j).fit(i2train_pca)
    poly_ocsvm = svm.OneClassSVM(gamma='scale', kernel='poly', nu=j, degree=2).fit(i2train_pca)

    L1 = ocsvm.predict(i3Atest_pca)
    L2 = rbf_ocsvm.predict(i3Atest_pca)
    L3 = poly_ocsvm.predict(i3Atest_pca)

    AnomResults = {-1: "Anomaly", 1: "OK"}

    # if model_predictions == -1:

    print(o3testClass.shape)
    L_arr = L1.reshape(-1, 1)
    print(L_arr.shape)
    vm_ocsvm = validate_model(L_arr, oClass_test_pca)
    vm_rbf_ocsvm = validate_model(L2.reshape(-1, 1), oClass_test_pca)
    vm_poly_ocsvm = validate_model(L3.reshape(-1, 1), oClass_test_pca)

    validate_model(L_arr, oClass_test_pca)
    if vm_ocsvm > best__ocsvm:
        best__ocsvm = vm_ocsvm
        best_comb = j
        print("OCSVM", best__ocsvm)

    if vm_rbf_ocsvm > best_rbf_ocsvm:
        best_rbf_ocsvm = vm_rbf_ocsvm
        best_comb = j
        print("rbf", best_rbf_ocsvm)

    if vm_poly_ocsvm > best_poly_ocsvm:
        best_poly_ocsvm = vm_poly_ocsvm
        best_comb = j
        print("poly", best_poly_ocsvm)

    ### Isolation Forest ###
    clf = IsolationForest(max_samples=100, random_state=0)
    clf.fit(i2train_pca)
    x = clf.predict(i3Atest_pca)
    print(x)

    max_samples = [100, 200, 300, 400]
    random_state = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    best = -1
    comb = list(itertools.product(max_samples, random_state))
    for a, b in comb:
        clf = IsolationForest(max_samples=a, random_state=b)
        clf.fit(i2train_pca)
        x = clf.predict(i3Atest_pca)
        print(x)
        print(a, b)
        result = validate_model(x.reshape(-1, 1), oClass_test_pca)
        print(validate_model(x.reshape(-1, 1), oClass_test_pca))
        if result > best:
            best = result
            best_comb = (a, b)
            print(best)

### Compare models ###




