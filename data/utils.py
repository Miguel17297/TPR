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

    return (np.sqrt(s / n))
