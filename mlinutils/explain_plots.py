import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="white")


def plot_feature_contrib(feature_names, contrib,
                         selected_ft=0,
                         crop_zeros=False,
                         show_values=True,
                         zero_element=0.,
                         title=None,
                         build_text_fn=lambda x: str(x)):
    """
        Renders a horizontal bar plot for the participation of each feature  to the final result
    :param feature_names: names of the explainable features that contribute to the model
    :param contrib: The numerical contribution of each feature (must be in the same order as feature_names)
    :param selected_ft: The number of features we want appearing in our plot (all features by default)
    :param crop_zeros: whether show the features that have a participation of zero (false by default)
    :param zero_element: the "zero" elements to be cropped
    :param show_values: whether show the feature value
    :param title: title of the plot
    :param build_text_fn: custom function to build the text on each bar
    :return: ax
    """

    assert len(contrib) == len(feature_names), "feature_names and participation should have the same length"
    assert selected_ft < len(feature_names), "selected_features out of range"

    if selected_ft == 0:
        selected_ft = len(contrib)

    nms = feature_names.copy()
    sorted_idx = np.argsort(np.abs(contrib))[::-1][:selected_ft]
    contrib = np.array([contrib[i] for i in sorted_idx])
    nms = [nms[i] for i in sorted_idx]
    color = [['b', 'r'][int(c < 0)] for c in contrib]

    if crop_zeros:
        contrib = list(filter(zero_element.__ne__, contrib))
        selected_ft = len(contrib)
        nms = nms[:selected_ft]
        color = color[:selected_ft]

    _, ax = plt.subplots(1, 1, figsize=(10, len(contrib)))
    ax.barh(range(selected_ft), contrib, color=color)
    ax.set_yticks(np.arange(selected_ft))
    ax.set_yticklabels(nms)
    ax.invert_yaxis()

    if show_values:
        for i, c in enumerate(contrib):
            # build text
            pos_x_index = 1 if c < 0.0 else 0
            ax.text((c, 0)[pos_x_index] + 0.05, i, build_text_fn(c), color='black')

    ax.set_title(title)
    return ax


def plot_pairwise_correlations(dataframe, title=None):
    """
    Plotting a diagonal correlation matrix computed from the given `dataframe`.
    """

    # Compute the correlation matrix
    corr = dataframe.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(9, 7))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    if title is not None:
        ax.set_title(title)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})


def plot_feature_importances(feature_names, feature_importances, importance_type, ax=None):
    """
        Plot relative feature importances.
    """
    import xgboost as xgb
    
    raw_importances = dict(zip(feature_names, feature_importances))
    
    # Normalize features relative to the maximum
    max_value = max(raw_importances.values())
    normalized_importances = {k: round((v/max_value)*100.0, 2) for k, v in raw_importances.items()}
    
    title = 'Feature Importances of {} Features using {}'.format(len(feature_names), importance_type)
    
    return xgb.plot_importance(normalized_importances, title=title, 
                               xlabel="relative importance", ylabel=None, 
                               height=0.9, grid=False, ax=ax)


def xgb_feature_importances(xgb_estimator, importance_type="weight", ax=None):
    """
        Utility method to plot feature importances from the given `xgb_estimator` using the importance_type.
    """
    def fix_feature_name(fn):
        return "x%s" % (int(fn[1:]) + 1)
    
    raw_importances = xgb_estimator.get_booster().get_score(importance_type=importance_type)
    
    feature_names = [fix_feature_name(k) for k in raw_importances.keys()] 
    feature_importances = raw_importances.values()

    return plot_feature_importances(feature_names, feature_importances, importance_type.upper(), ax)


def xgb_plot_comparative_feature_importances(xgb_estimator, fig_size=(20,5)):
    """
        Plots a 3-columned figure for each type of XGBoost feature importance from the give `xgb_estimator`.
    """
    _, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=fig_size)  
    xgb_feature_importances(xgb_estimator, ax=ax1, importance_type="weight")
    xgb_feature_importances(xgb_estimator, ax=ax2, importance_type="gain")
    xgb_feature_importances(xgb_estimator, ax=ax3, importance_type="cover")


class AbstractLocalExplainer:
    """
        Base class for plotting local contributions
    """
    
    def get_features_contribution(self, instance_as_series):
        return None
    
    def get_local_prediction(self, features_contribution):
        return None
    
    def get_base_value(self):
        return None
    
    def get_name(self):
        return None


class ShapelyExplainer(AbstractLocalExplainer):
    
    def __init__(self, kernel_explainer):
        self.kernel_explainer = kernel_explainer
    
    def get_features_contribution(self, instance_as_series):
        return instance_as_series.index, self.kernel_explainer.shap_values(instance_as_series)
    
    def get_local_prediction(self, features_contribution):
        return features_contribution.sum() + self.kernel_explainer.expected_value
    
    def get_base_value(self):
        return self.kernel_explainer.expected_value
    
    def get_name(self):
        return "Shapely" 


class LimeExplainer(AbstractLocalExplainer):
    
    def __init__(self, lime_explainer, predict_func, num_samples=5000):
        self.lime_explainer = lime_explainer
        self.predict_func = predict_func
        self.num_samples = num_samples
    
    def get_features_contribution(self, instance_as_series):
        self.exp = self.lime_explainer.explain_instance(instance_as_series, 
                                                        self.predict_func, 
                                                        num_features=instance_as_series.shape[0], 
                                                        num_samples=self.num_samples)
        
        # LIMEs use a Ridge Regression as local model fitted over scaled data.
        # So that we have to scale the input data in order to compute
        # feature contribution = weight * input value (scaled)
        scaler = self.lime_explainer.scaler
        scaled_instance = (instance_as_series - scaler.mean_) / scaler.scale_
        
        local_exp = self.exp.as_map()[1]
        coefs = [x[1] for x in local_exp]
        ids = [x[0] for x in local_exp]
        
        weights = coefs * scaled_instance[ids]
        feature_names = scaled_instance[ids].index
        
        return feature_names, weights
    
    def get_local_prediction(self, features_contribution):
        return self.exp.local_pred[0]
    
    def get_base_value(self):
        return self.exp.intercept[1]
    
    def get_name(self):
        return "LIME" 


class TreeinterpreterExplainer(AbstractLocalExplainer):
    
    def __init__(self, xgboost_estimator):
        self.estimator = xgboost_estimator.get_booster()
        self.biais = -1

    def get_features_contribution(self, instance_as_dataframe):
        from eli5 import explain_prediction_df
        
        explain_df = explain_prediction_df(self.estimator, instance_as_dataframe.values)
        
        # separate the bias from features contribution
        mask_bias = explain_df.feature == "<BIAS>"
        
        self.biais = explain_df[mask_bias].weight.values[0]
        
        feature_names = explain_df[~mask_bias].feature.values
        feature_contributions = explain_df[~mask_bias].weight.values
        
        return feature_names, feature_contributions
    
    def get_local_prediction(self, feature_contributions):
        return sum(feature_contributions) + self.biais
    
    def get_name(self):
        return "Treeinterpreter (ELI5)" 
    
    def get_base_value(self):
        return self.biais


def break_down_plot(instance, local_explainer, show_base_bar=False, 
                    show_final_pred_bar=False, show_values=False, ax=None):
    """
        Plots a break down for local contributions
    """

    assert isinstance(local_explainer, AbstractLocalExplainer),  "expect an instance of AbstractLocalExplainer"
    
    feature_names, features_contribuition = local_explainer.get_features_contribution(instance)
    
    break_down_df = pd.DataFrame(np.asarray(features_contribuition).reshape(-1,1), 
                                 columns=["weight"], 
                                 index=feature_names)
    break_down_df.sort_values(by='weight', inplace=True)
    
    base_value = local_explainer.get_base_value()
    if show_base_bar:
        base_value_row = pd.DataFrame([base_value], columns=["weight"], index=["base value"])
        break_down_df = pd.concat([base_value_row, break_down_df])
    
    final_prediction = local_explainer.get_local_prediction(features_contribuition)
    
    if show_final_pred_bar:
        final_pred_row = pd.DataFrame([final_prediction], columns=["weight"], index=["final prediction"])
        break_down_df = pd.concat([break_down_df, final_pred_row])
    
    # setup colors
    colors = ['#1f77b4' if x > 0 else '#d62728' for x in break_down_df.weight]
    if show_base_bar:
        colors[0] = "grey"
        
    if show_final_pred_bar:
        colors[-1] = "black"
    
    # make title
    title = 'Break down for local explanation using ' + local_explainer.get_name()
    title += "\n\nSum of local weights: " + str(round(final_prediction, 5))
    
    ax = break_down_df.plot(kind='bar', title=title, legend=False, color=[colors],  width=0.9, ax=ax)
    
    # adjust y limits
    y_lims = ax.get_ylim()
    ax.set_ylim(y_lims[0] - .05, y_lims[1] + .05)
    
    if show_values:
        # set individual bar lables 
        for i in ax.patches:
            bar_height = i.get_height()
            bar_y_pos = (bar_height + 0.025) if bar_height > 0 else (bar_height - 0.09)
            bar_value = ("%+.2f" % bar_height) if abs(bar_height) > 0.01 else ""
            ax.text(i.get_x() - .03, bar_y_pos, bar_value, fontsize=8, color='black')
        
    return ax
