from collections import Counter
from itertools import product

import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import utils as ut
from predict.instance_feature import load_instance
from text.abstract_graph.feature_utils import load_event_feat, load_vocab_mappings


def PriceChangeDiscretizer(y):
    yt = y.copy()

    pt1 = [3, 2, 1, -1, -2]
    pt2 = [1, 1, 1, -1, -1]

    pt = pt2

    yt[y >= 0.025] = pt[0]
    yt[(y >= 0.01) & (y < 0.025)] = pt[1]
    yt[(y >= 0.0) & (y < 0.01)] = pt[2]
    yt[(y >= -0.01) & (y < 0.0)] = pt[3]
    yt[y < -0.01] = pt[4]

    return yt


def random_forest_classification():
    X, Y = load_instance()
    n, d = X.shape
    _, k = Y.shape
    R = np.zeros((d, k))
    for i, j in product(range(k), repeat=2):
        x, y = X[:, i], Y[:, j]
        cond = ~(np.isnan(x) | np.isnan(y))
        R[i, j] = pearsonr(x[cond], y[cond])[0]

    id2events, event_feats = load_event_feat(load_vocab_mappings())

    models = []
    for i in [
        # 0, 1, 4
        2
    ]:
        index = ~np.isnan(Y[:, i])

        y_i = PriceChangeDiscretizer(Y[index, i])
        X_i = X[index]

        # for i in range(5, d):
        #     x = X_i[:, i]
        #     print("******"*20)
        #     print("Event {} {}".format(i - 4, id2events[str(i - 4)]))
        #     for j in sorted(ut.distinct(y_i)):
        #         sub_x = x[y_i == j]
        #         print("Label {}: {}".format(
        #             j, sorted(
        #                 [(k, round(100*v/sub_x.shape[0], 2) if sub_x.shape[0] > 0 else 0) for k, v in
        #                  Counter(sub_x).items()],
        #                 key=lambda tpl: tpl[0]
        #             )))

        X_train, X_test, y_train, y_test = train_test_split(X_i, y_i, test_size=0.2)

        selector = SelectKBest(mutual_info_classif, k=30)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        for d in [5, 7, 10, 13]:
            mod = RandomForestClassifier(n_estimators=10, max_depth=d)
            mod.fit(X_train, y_train)
            y_train_pred = mod.predict(X_train)
            y_test_pred = mod.predict(X_test)
            train_acc = balanced_accuracy_score(y_train, y_train_pred)
            test_acc = balanced_accuracy_score(y_test, y_test_pred)
            print("For classification task %i: Depth %i; Train Acc: %.4f; Test Acc: %.4f;"%(i, d, train_acc, test_acc))
            # print("For classification task %i: Test Acc: %.4f;"%(i, test_acc))
            models.append(mod)

    return R


def random_forest_regression():
    return


if __name__ == '__main__':
    random_forest_classification()
