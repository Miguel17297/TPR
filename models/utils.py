import sys
import matplotlib.pyplot as plt
import seaborn
import numpy as np


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
    """
    Validates model performance
    :param model_predictions: predictions (anomaly, normal) made by a model for a given data
    :param real_values: true data classification (equivale Classes[o3testClass[i][0]])
    """

    assert model_predictions.shape == real_values.shape

    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(len(model_predictions)):

        if model_predictions[i][0] == -1 and real_values[i][0] == -1:  # true positive
            tp = tp + 1
        elif model_predictions[i][0] == -1 and real_values[i][0] == 1:  # false positives
            fp = fp + 1
        elif model_predictions[i][0] == 1 and real_values[i][0] == 1:  # true negative
            tn = tn + 1
        else:
            fn = fn + 1

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    recall_den = (tp + fn)
    recall = tp / recall_den if recall_den > 0 else 0

    precision_den = (tp + fp)
    precision = tp / precision_den if precision_den > 0 else 0

    f1_score_den = (recall + precision)
    f1_score = 2 * (recall * precision) / f1_score_den if f1_score_den > 0 else 0

    print(f"\n\tTrue Positives: {tp}, False Positives: {fp}, True Negatives: {tn}, False Negatives: {fn}")
    print(f"\tAcurracy: {accuracy}")
    print(f"\tRecall: {recall}")
    print(f"\tPrecision: {precision}")
    print(f"\tF1-Score: {f1_score}")

    labels = ["Anomaly", "Normal"]
    results = np.array([tp, fp, tn, fn])
    results = results.reshape(2, 2)

    # seaborn.set(font_scale=1.4)
    # ax = seaborn.heatmap(results , annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})
    # ax.set_xticklabels(labels)
    # ax.set_yticklabels(labels)

    # ax.set(xlabel="Real Values", ylabel="Predicted Values")
    # plt.show()

    return f1_score


def dataset_division(data, percentage=0.7):
    pn = int(len(data) * percentage)

    train = data[: pn:]
    test = data[pn:, :]

    return train, test


## -- 11 -- ##
def distance(c, p):
    s = 0
    n = 0
    for i in range(len(c)):
        if c[i] > 0:
            s += np.square((p[i] - c[i]) / c[i])
            n += 1

    return np.sqrt(s / n) if n>0 else 0