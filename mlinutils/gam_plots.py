import matplotlib.pyplot as plt
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from mlinutils.explain_plots import plot_feature_contrib
from pygam import GAM


class GamSckitWrapper(BaseEstimator, ClassifierMixin):
    """
        Utility class for wrapping a base estimator and pass it to YellowBrick vizualizers.
    """
    
    def __init__(self, gam_model, classes):

        self.gam_model = gam_model
        self.classes_ = classes
    
    def fit(self, X, y=None):
        """
        backed model is already fitted then just return it
        """
        return self
    
    def predict_proba(self, X):
        
        pos_probas = self.gam_model.predict_proba(X).reshape(-1, 1)
        probas = np.hstack((1 - pos_probas, pos_probas))
        
        return probas
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        return self.gam_model.predict(X)
        
    def name(self):
        return self.gam_model.__class__.__name__


def local_regression_plot(sample, feature_names, gam, target_unity, selected_ft=10):

    prediction = gam.predict([sample])[0]
    contrib = gam.partial_dependence([sample])[0]
    base_value = gam.partial_dependence(feature='intercept')[0][0]

    print(f"The prediction is around {prediction:,.2f} {target_unity}")
    print(f"Base value (intercept) accounts for {base_value:,.2f} {target_unity}")

    def text_builder(v): return f"{v:.0f} {target_unity}"

    return plot_feature_contrib(feature_names=feature_names,
                                contrib=contrib,
                                build_text_fn=text_builder,
                                selected_ft=selected_ft,
                                crop_zeros=True,
                                title="Features contribution to the final decision")


def local_logistic_plot(sample, feature_names, gam_logistic, classes_label=["No", "Yes"], selected_ft=10):

    log_odds = gam_logistic.partial_dependence([sample])[0]
    odds_contrib = np.exp(log_odds)

    # converts in "times" scale
    odds_times = np.where(odds_contrib < 1, np.divide(-1, odds_contrib), odds_contrib)

    prediction_prob_true = gam_logistic.predict_proba([sample])[0]
    pos_label = classes_label[1]
    print(f'Probability of "{pos_label}" is: {prediction_prob_true:.2f}')

    def text_builder(odds_rate):
        adjusted_odd_rate = odds_rate
        if odds_rate < 0:
            return f"{-1*adjusted_odd_rate:.2f}x less likely to {pos_label}"
        else:
            return f"{adjusted_odd_rate:.2f}x more likely to {pos_label}"

    return plot_feature_contrib(feature_names=feature_names,
                                contrib=odds_times,
                                build_text_fn=text_builder,
                                selected_ft=selected_ft,
                                crop_zeros=True,
                                zero_element=1.,
                                title="Features contribution to the final decision")
    

def global_explanation_plot(gam, feature_names, number_cols=4):

    number_lines = len(feature_names)//2
    if (len(feature_names) % 2) != 0:
        number_lines += 1
        
    fig, axs = plt.subplots(number_lines, number_cols)
    fig.set_size_inches(20, 2 * number_lines * number_cols)
    titles = feature_names
    
    xx = GAM.generate_X_grid(gam)
    for j, sub_axs in enumerate(axs):
        for i, ax in enumerate(sub_axs):
            if number_cols*j+i >= len(titles):
                ax.remove()
            else:
                pdep, confi = gam.partial_dependence(xx, feature=number_cols * j + i, width=.95)
                ax.plot(xx[:, 0], pdep, LineWidth=3)
                ax.plot(xx[:, 0], confi[0][:, 0], c='grey', ls='--', alpha=0.6)
                ax.plot(xx[:, 0], confi[0][:, 1], c='grey', ls='--', alpha=0.6)
                ax.set_title(titles[number_cols * j + i], pad=10, fontdict={'fontsize': 20, 'fontweight': 'bold'})