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
from scipy.stats import multivariate_normal
from sklearn import svm
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
    tp, fp, tn, fn = 0, 0, 0,0

    for i in range(len(model_predictions)):

        if model_predictions[i] == 1 and real_values[1] == 1: # true positive
            tp = tp + 1
        elif model_predictions[i] == 1 and real_values[1] == -1: # false positives
            fp = fp + 1
        elif  model_predictions[i] == -1 and real_values[1] == -1: # true negative
            tn = tn + 1
        else:
            fn = fn + 1

    accuracy = (tp+tn) / (tp+tn+fp+fn)
    recall = tp / (tp+fn)
    precision = tp /(tp + fp)
    f1_score = 2*(recall * precision) / (recall + precision)

    print(f"True Positives: {tp}, False Positives: {fp}, True Negatives: {tn}, False Negatives: {fp} \n")
    print(f"Acurracy: {accuracy} \n")
    print(f"Recall: {recall} \n")
    print(f"Precision: {precision} \n")
    print(f"F1-Score: {f1_score} \n")

    labels = ["Anomaly", "Normal"]
    results = np.array([tp, fp, tn, fn])

    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(results, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set(ylabel="Real Values", xlabel="Predicted Values")
    plt.show()




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

oClass_bot3= np.ones((len(features_bot3), 1)) * 1
print(features_bot3.shape)
# oClass_bot1 = np.ones((len(features_bot3), 1)) * 0

oClass_normal = np.ones((len(features_normal), 1)) * 0
print(features_normal.shape)



features = np.vstack((features_normal, features_bot3))

oClass= np.vstack(( oClass_normal,oClass_bot3))

Obs, nFea = features_bot3.shape
print('Train Silence Features Size:', features.shape)

plt.figure(1)
plotFeatures(features, oClass, 0,nFea-1)

# sys.exit(0)


## -- 3 -- ##
#:i
percentage = 0.7
# pB = int(len(features_bot1) * percentage)
# trainfeatures_bot1 = features_bot1[:pB, :]
# pYT = int(len(features_bot2) * percentage)
# trainfeatures_bot2 = features_bot2[:pYT, :]
pb3 = int(len(features_bot3) * percentage)
trainfeatures_bot3 = features_bot3[:pb3 :]
Pn = int(len(features_normal)*percentage)
trainfeatures_normal = features_normal[: Pn:]
test_features_normal = features_normal[Pn: ,:]
test_features_bot3 = features_bot3[pb3: ,:]

# i2train = np.vstack((trainfeatures_bot1, trainfeatures_bot2))
# o2trainClass = np.vstack((oClass_browsing[:pB], oClass_yt[:pYT]))
oNormal_trainClass = oClass_normal[: Pn:]

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
centroids=np.mean(trainfeatures_normal,axis=0)
print('All Features Centroids:\n', centroids)

i3Atest = np.vstack((test_features_bot3,test_features_normal))
# i3Atest= scaler.transform(trainfeatures_bot3)

AnomalyThreshold = [5,10,3,7]

o3testClass = np.vstack((oClass_bot3,oClass_normal))
print(o3testClass)

for j in  AnomalyThreshold:
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

        print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f}] -> Result -> {}'.format(i,Classes[o3testClass[i][0]],*dists,results[i]))
        # print(i)
        # print(Classes[o3testClass[i][0]])
        # print(*dists)
        # print(results[i])
            


## -- 8 -- ##

print('\n-- Anomaly Detection based on One Class Support Vector Machines--')
i2train = trainfeatures_normal
i3Atest = np.vstack((test_features_bot3,test_features_normal ))

nu = [0.1,0.4,0.5,0.8]

for j in nu:
    ocsvm = svm.OneClassSVM(gamma='scale', kernel='linear', nu=j).fit(i2train)
    rbf_ocsvm = svm.OneClassSVM(gamma='scale', kernel='rbf', nu=j).fit(i2train)
    poly_ocsvm = svm.OneClassSVM(gamma='scale', kernel='poly', nu=j, degree=2).fit(i2train)

    L1 = ocsvm.predict(i3Atest)
    L2 = rbf_ocsvm.predict(i3Atest)
    L3 = poly_ocsvm.predict(i3Atest)

    AnomResults = {-1: "Anomaly", 1: "OK"}

    nObsTest, nFea = i3Atest.shape
    for i in range(nObsTest):
        print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i, Classes[
            o3testClass[i][0]], AnomResults[L1[i]], AnomResults[L2[i]], AnomResults[L3[i]]))



