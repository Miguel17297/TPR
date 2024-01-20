import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import sys
import warnings
from sklearn.preprocessing import MaxAbsScaler
from scipy.stats import multivariate_normal
from sklearn import svm
from scapy.all import sniff
 
 
warnings.filterwarnings('ignore')


def waitforEnter(fstop=False):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else :
            input("Press ENTER to continue.")
            
def plotFeatures(features,oClass,f1index=0,f2index=0):
    nObs,nFea = features.shape
    cObs,cFea = oClass
    colors = ['b','r','g','y']
    for i in range(nObs):
        if i < cObs:
            plt.plot(features[i,f1index],features[i,f2index],colors[cFea[i]]+'o')
    plt.show()
    waitforEnter()
    
def distance (x,y):
    return (np.sqrt(np.sum(np.quare(x-y))))

#def Stats()


classes = {0:'Client',1:"Attacker"}
nfig =1

#features_c1 = np.loadtxt(linkedIn)
#features_c2 = np.loadtxt(tripadvisor)
#features_c3 = np.loadtxt(taobao)
#features_c4 = np.loadtxt(attacker)

oClass_client = np.ones((len(features_c1),len(features_c2),1))
oClass_attacker = np.ones((len(features_c4),1))*1
oClass_client_test = np.ones((len(features_c3),1))*0

features = np.vstack((features_c1,features_c2,features_c3))
oClass = np.vstack((oClass_client,oClass))

print('Train Stats Features Size:',features.shape)
print('Classes Size: ', oClass.shape)

# features_c1S=np.loadtxt()
# features_c2S=np.loadtxt(
# features_c3S=np.loadtxt()
# features_c4S=np.loadtxt()



##Feature training
# build train features for normal behavior
trainFeaturesClient = np.vstack((features_c1,features_c2))
trainFeaturesClientS = np.vstack((features_c1S,features_c2S))
allTrainFeaturesClient = np.vstack((trainFeaturesClient,trainFeaturesClientS))
trainClassClient = oClass_client

# build train features for abnormal behavior
trainFeaturesAttacker = features_c4
trainFeaturesAttackerS = features_c4S
allTestFeaturesAttacker = np.vstack((trainFeaturesAttacker,trainFeaturesAttackerS))
testClassAttacker = oClass_attacker


## feature normlization

from sklearn.preprocessing import MaxAbsScaler

trainScaler = MaxAbsScaler().fit(allTrainFeaturesClient)
trainFeaturesN = trainScaler.transform(allTrainFeaturesClient)

##Normalize test classes with train Scalers

AtestFeaturesNA = trainScaler.transform(allTestFeaturesAttacker)
AtestFeaturesNC = trainScaler.transform(allTestFeaturesClient)

#Componets Analysis

from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit(trainFeaturesN)









       
 
 
 