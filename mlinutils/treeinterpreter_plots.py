from mlinutils.explain_plots import plot_feature_contrib


def regressor_break_down_plot(prediction, bias, contributions, sample, unity=None):
    print(f"Predicted: {prediction[0]:.0f} {unity}")
    print(f"Base value (bias) accounts for: {bias[0]:.0f} {unity}")
    readable_features_names=[f"{f}={v:.2f}" for f, v in zip(sample.index, sample.values)]
    return plot_feature_contrib(feature_names=readable_features_names,
                                contrib=contributions[0],
                                selected_ft=10,
                                crop_zeros=True,
                                title="Features contribution calculated by TreeInterpreter",
                                build_text_fn=lambda x: f"{x:.0f} {unity}");


def classifier_break_down_plot(prediction, bias, contributions, sample):
    print(f"Probability of default: {100*prediction[0][1]:.2f}%")
    print(f"Base value (bias) accounts for: {100*bias[0][1]:.2f}%")
    readable_features_names=[f"{f}={v:.2f}" for f, v in zip(sample.index, sample.values)]
    plot_feature_contrib(feature_names=readable_features_names, 
                         contrib=contributions[0][:,1] * 100,
                         selected_ft=10,
                         crop_zeros=True,
                         title="Features contribution calculated by TreeInterpreter",
                         build_text_fn=lambda x: f"{x:.2f}%");
