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


def waitforEnter(fstop=False):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")


## -- 3 -- ##
def plotFeatures(features, oClass, f1index=0, f2index=1):
    nObs, nFea = features.shape
    colors = ['b', 'g', 'r']
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
Classes = {0: 'Browsing', 1: 'YouTube', 2: 'Mining'}
plt.ion()
nfig = 1

## -- 2 -- ##
features_browsing = np.loadtxt("BrowsingAllF.dat")
features_yt = np.loadtxt("YouTubeAllF.dat")
features_mining = np.loadtxt("MiningAllF.dat")

oClass_browsing = np.ones((len(features_browsing), 1)) * 0
oClass_yt = np.ones((len(features_yt), 1)) * 1
oClass_mining = np.ones((len(features_mining), 1)) * 2

features = np.vstack((features_yt, features_browsing, features_mining))
oClass = np.vstack((oClass_yt, oClass_browsing, oClass_mining))

print('Train Silence Features Size:', features.shape)
plt.figure(2)
plotFeatures(features, oClass, 0, 6)
plt.figure(3)
plotFeatures(features, oClass, 2, 8)
plt.figure(4)
plotFeatures(features, oClass, 5, 15)

## -- 3 -- ##
#:i
percentage = 0.5
pB = int(len(features_browsing) * percentage)
trainFeatures_browsing = features_browsing[:pB, :]
pYT = int(len(features_yt) * percentage)
trainFeatures_yt = features_yt[:pYT, :]
pM = int(len(features_mining) * percentage)
trainFeatures_mining = features_mining[:pYT, :]

i2train = np.vstack((trainFeatures_browsing, trainFeatures_yt))
o2trainClass = np.vstack((oClass_browsing[:pB], oClass_yt[:pYT]))

#:ii
i3Ctrain = np.vstack((trainFeatures_browsing, trainFeatures_yt, trainFeatures_mining))
o3trainClass = np.vstack((oClass_browsing[:pB], oClass_yt[:pYT], oClass_mining[:pM]))

#:iii
testFeatures_browsing = features_browsing[pB:, :]
testFeatures_yt = features_yt[pYT:, :]
testFeatures_mining = features_mining[pM:, :]

i3Atest = np.vstack((testFeatures_browsing, testFeatures_yt, testFeatures_mining))
o3testClass = np.vstack((oClass_browsing[pB:], oClass_yt[pYT:], oClass_mining[pM:]))

## -- 7 -- ##

i2train = np.vstack((trainFeatures_browsing, trainFeatures_yt))
# scaler = MaxAbsScaler().fit(i2train)
# i2train=scaler.transform(i2train)

centroids = {}
for c in range(2):  # Only the first two classes
    pClass = (o2trainClass == c).flatten()
    centroids.update({c: np.mean(i2train[pClass, :], axis=0)})
print('All Features Centroids:\n', centroids)

i3Atest = np.vstack((testFeatures_browsing, testFeatures_yt, testFeatures_mining))
# i3Atest=scaler.transform(i3Atest)

AnomalyThreshold = 10

print('\n-- Anomaly Detection based on Centroids Distances --')
nObsTest, nFea = i3Atest.shape
results = np.ones(nObsTest)
for i in range(nObsTest):
    x = i3Atest[i]
    dists = [distance(x, centroids[0]), distance(x, centroids[1])]
    if min(dists) > AnomalyThreshold:
        results[i] = -1
    else:
        results[i] = 1

## -- 8 -- ##

print('\n-- Anomaly Detection based on One Class Support Vector Machines--')
i2train = np.vstack((trainFeatures_browsing, trainFeatures_yt))
i3Atest = np.vstack((testFeatures_browsing, testFeatures_yt, testFeatures_mining))

nu = 0.1
ocsvm = svm.OneClassSVM(gamma='scale', kernel='linear', nu=nu).fit(i2train)
rbf_ocsvm = svm.OneClassSVM(gamma='scale', kernel='rbf', nu=nu).fit(i2train)
poly_ocsvm = svm.OneClassSVM(gamma='scale', kernel='poly', nu=nu, degree=2).fit(i2train)

L1 = ocsvm.predict(i3Atest)
L2 = rbf_ocsvm.predict(i3Atest)
L3 = poly_ocsvm.predict(i3Atest)

AnomResults = {-1: "Anomaly", 1: "OK"}

nObsTest, nFea = i3Atest.shape
for i in range(nObsTest):
    print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i, Classes[
        o3testClass[i][0]], AnomResults[L1[i]], AnomResults[L2[i]], AnomResults[L3[i]]))



