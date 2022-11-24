import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from torch.utils.data import Subset


def plot_calib_curve(y_true, y_pred_prob, y_pred_prob_calib=None, bins=20):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=bins)

    ax[0].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax[0].plot(prob_pred, prob_true, "s-", label="Non-calibrated model")
    ax[0].set_ylabel("Fraction Positives")
    ax[0].set_xlabel("Mean predicted probability")

    if y_pred_prob_calib is not None:
        prob_true, prob_pred_calib = calibration_curve(y_true, y_pred_prob_calib, n_bins=bins)
        ax[0].plot(prob_true, prob_pred_calib, "s-", label="Calibrated model")

    ax[0].set_ylim([-0.05, 1.05])
    ax[0].legend(loc="lower right")

    ax[1].hist(y_pred_prob, range=(0, 1), bins=20, histtype="step", lw=2, label="Non-calibrated model")

    if y_pred_prob_calib is not None:
        ax[1].hist(y_pred_prob_calib, range=(0, 1), bins=20, histtype="step", lw=2, label="Calibrated model")
        ax[1].legend()

    ax[1].set_xlabel("Mean predicted probability")
    ax[1].set_ylabel("Count")
    return


class CalibratedModel:
    def __init__(self, model):
        self.model = model
        self.fitted = False
        self.calibrator = LogisticRegression()

    def fit(self, X, y, return_model_probs=False):
        pred_prob = self.model.predict_proba(X)
        y_multi = np.array(y) + 1
        self.calibrator.fit(pred_prob, y_multi)
        self.fitted = True

        if return_model_probs:
            return pred_prob

    def predict_proba(self, X):
        if not self.fitted:
            raise Exception("Calibrator not fitted!")

        model_preds = self.model.predict_proba(X)
        return self.calibrator.predict_proba(model_preds)

    def predict(self, X):
        if not self.fitted:
            raise Exception("Calibrator not fitted!")

        model_preds = self.model.predict_proba(X)
        return self.calibrator.predict(model_preds)


def one_hot_to_1d(arr):
    return np.argmax(arr, axis=1)


def one_hot_transform(labels_1d, unique_label_count="infer"):
    if unique_label_count == 'infer':
        unique_label_count = labels_1d.max() + 1

    encoded = np.zeros((len(labels_1d), unique_label_count), dtype=int)
    encoded[np.arange(len(labels_1d)), labels_1d] = 1
    return encoded


def predict(clf, dataset, boundary):
    pred_probs = clf.predict_proba(dataset)

    if type(boundary) == tuple:
        return predict_flexible_boundary(pred_probs, boundary)

    else:
        return predict_boundary(pred_probs, boundary)


def predict_boundary(pred_probs, boundary):
    max_prob_idx = np.argmax(pred_probs, axis=1)
    max_probs = np.take_along_axis(pred_probs, max_prob_idx.reshape(-1, 1), axis=1).reshape(-1)

    predictions = np.zeros(len(pred_probs), dtype=int)
    predictions[max_prob_idx > 0] = 2
    predictions[max_probs < boundary] = 1
    return predictions


def predict_flexible_boundary(pred_probs, boundary):
    b_0, b_2 = boundary
    pred_probs = np.insert(pred_probs, 1, np.zeros(len(pred_probs)), axis=1)
    # Set predicted probabilities to zero for non-maximal probs
    pred_probs[~(pred_probs == np.max(pred_probs, keepdims=True, axis=1))] = 0
    # Set predicted probabilities not reaching respective boundary to zero
    pred_probs[pred_probs[:, [0, 1, 2]] < np.array([b_0, 0, b_2])] = 0
    # Predicting neutral category where pos/neg not reached boundary
    pred_probs[(pred_probs == 0).all(axis=1), 1] = 1
    pred_probs[pred_probs > 0] = 1
    return one_hot_to_1d(pred_probs.astype(int))


def compute_decision_boundary(clf, dataset, boundaries, scorer=f1_score, scorer_args={"average": "macro"},
                              return_scores=False, flexible=False):
    pred_probs = clf.predict_proba(dataset)
    labels = [y for _, y in dataset]
    labels = one_hot_transform(np.array(labels))
    scores = []
    for b in boundaries:
        if flexible:
            preds = predict_flexible_boundary(pred_probs, b)

        else:
            preds = predict_boundary(pred_probs, b)

        preds = one_hot_transform(preds, unique_label_count=labels.shape[1])
        scores.append(scorer(labels, preds, **scorer_args))
    if return_scores:
        return boundaries[np.argmax(scores)], scores
    else:
        return boundaries[np.argmax(scores)]


def dec_bound_opt(clf, dataset, boundaries, flexible=False, scorer=f1_score, scorer_args={"average": "macro"}):
    cv_split = KFold(shuffle=True, n_splits=5)

    test_scores = []

    for train_split, test_split in cv_split.split(dataset):
        train, test = Subset(dataset, train_split), Subset(dataset, test_split)
        dec_bound = compute_decision_boundary(clf, train, boundaries, flexible=flexible, scorer=scorer,
                                              scorer_args=scorer_args)
        test_pred = predict(clf, test, dec_bound)
        test_true = [y for _, y in test]
        test_scores.append(scorer(test_true, test_pred, **scorer_args))

    return test_scores
