import numpy as np
from mlinutils.explain_plots import plot_feature_contrib


def global_explanation_plot(feature_names, linear_model, selected_ft=0):
    # extract coefficients rom the linear model
    coefs = linear_model.coef_
    if coefs.shape[0] == 1:
        coefs = coefs[0]

    plot_feature_contrib(feature_names=feature_names, contrib=coefs, selected_ft=selected_ft, crop_zeros=True,
                         show_values=False, title="Global features importance")


def local_regression_plot(sample, feature_names, regressor, target_unity, selected_ft=0):
    sample_as_array = np.array(sample)
    coefs = regressor.coef_
    base_value = regressor.intercept_

    prediction = regressor.predict([sample_as_array])[0]
    contrib = [c * f for (c, f) in zip(coefs, sample_as_array)]

    print(f"The prediction is around {prediction:,.2f} {target_unity}")
    print(f"Base value (intercept) accounts for {base_value:,.2f} {target_unity}")

    def text_builder(v): return f"{v:.0f} {target_unity}"

    return plot_feature_contrib(feature_names=feature_names,
                                contrib=contrib,
                                build_text_fn=text_builder,
                                selected_ft=selected_ft,
                                crop_zeros=True,
                                title="Features contribution to the final decision")


def local_logistic_plot(sample, feature_names, logistic_model, classes_label=["No", "Yes"], selected_ft=10):

    log_odds = np.multiply(logistic_model.coef_[0], sample)
    odds_contrib = np.exp(log_odds)

    # converts in "times" scale
    odds_times = np.where(odds_contrib < 1, np.divide(-1, odds_contrib), odds_contrib)

    prediction_prob_true = logistic_model.predict_proba([sample])[0][1]
    pos_label = classes_label[1]
    print(f'Probability of "{pos_label}" is: {prediction_prob_true:.2f}')

    def text_builder(odds_rate):
        if odds_rate < 0:
            return f"{-1*odds_rate:.2f}x less likely to {pos_label}"
        else:
            return f"{odds_rate:.2f}x more likely to {pos_label}"

    return plot_feature_contrib(feature_names=feature_names,
                                contrib=odds_times,
                                build_text_fn=text_builder,
                                selected_ft=selected_ft,
                                crop_zeros=True,
                                zero_element=1.,
                                title="Features contribution to the final decision")
