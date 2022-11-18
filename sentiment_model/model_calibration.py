import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression


def plot_calib_curve(y_true, y_pred_prob, y_pred_prob_calib=None, bins=20):
    fig, ax = plt.subplots(2,1, sharex=True, figsize=(10,10), gridspec_kw={'height_ratios':[3,1]})
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=bins)

    ax[0].plot([0,1], [0, 1], "k:", label="Perfectly calibrated")
    ax[0].plot(prob_pred, prob_true, "s-", label="Non-calibrated model")
    ax[0].set_ylabel("Fraction Positives")
    ax[0].set_xlabel("Mean predicted probability")

    if y_pred_prob_calib is not None:
        prob_true, prob_pred_calib = calibration_curve(y_true, y_pred_prob_calib, n_bins=bins)
        ax[0].plot(prob_true, prob_pred_calib, "s-", label="Calibrated model")

    ax[0].set_ylim([-0.05, 1.05])
    ax[0].legend(loc="lower right")

    ax[1].hist(y_pred_prob, range=(0,1), bins=20, histtype="step", lw=2, label="Non-calibrated model")

    if y_pred_prob_calib is not None:
        ax[1].hist(y_pred_prob_calib, range=(0,1), bins=20, histtype="step", lw=2, label="Calibrated model")
        ax[1].legend()

    ax[1].set_xlabel("Mean predicted probability")
    ax[1].set_ylabel("Count")
    return


class CalibratedModel:
    def __init__(self, model):
        self.model = model
        self.fitted = False
        self.calibrator = LogisticRegression()

    def fit(self, X, y):
        pred_prob = self.model.predict_proba(X)
        y_multi = np.array(y) + 1
        self.calibrator.fit(pred_prob, y_multi)
        self.fitted = True

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
