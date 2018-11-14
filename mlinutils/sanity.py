import matplotlib.pyplot as plt


from yellowbrick.regressor import ResidualsPlot, PredictionError
from yellowbrick.classifier import ROCAUC

from sklearn.metrics import r2_score

import seaborn as sns
sns.set(style="white")


def regression_sanity_check(model,X_train, X_test,y_train,y_test):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,10))  
    plt.sca(ax1)
    visualizer = ResidualsPlot(model, ax=ax1)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    plt.sca(ax2)
    visualizer2 = PredictionError(model, ax=ax2)
    visualizer2.fit(X_train, y_train)
    visualizer2.score(X_test, y_test)
    visualizer.finalize()
    visualizer2.poof()  


def classification_sanity_check(model, X_train, X_test, y_train, y_test, classes=None):
    visualizer = ROCAUC(model, micro=False, macro=False, classes=classes)

    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.poof()


def gam_regression_sanity_check(gam, X_train, X_test, y_train, y_test):
    residuals_train = gam.deviance_residuals(X_train, y_train)
    y_train_p = gam.predict(X_train)
    r2_train = r2_score(y_train, y_train_p)

    residuals_test = gam.deviance_residuals(X_test, y_test)
    y_test_p = gam.predict(X_test)
    r2_test = r2_score(y_test, y_test_p)

    label_train = "Train $R^2 = {:0.3f}$".format(r2_train)
    label_test = "Test $R^2 = {:0.3f}$".format(r2_test)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    plt.sca(ax1)
    ax1.set_aspect('equal', adjustable='box')
    plt.scatter(y_train_p, residuals_train)
    plt.scatter(y_test_p, residuals_test, color='g')
    plt.legend([label_train, label_test])
    plt.axhline(y=0, color='black', alpha=0.5)
    plt.title("Residuals of GAM model")
    plt.xlabel("Predicted value")
    plt.ylabel("Residuals")

    plt.sca(ax2)
    plt.scatter(y_test, y_test_p, label=label_test)
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title("Prediction error for GAMs")
    xlim = ax2.get_xlim()
    ylim = ax2.get_ylim()
    # Find the range that captures all data
    bounds = (min(ylim[0], xlim[0]), max(ylim[1], xlim[1]))

    # Reset the limits
    ax2.set_xlim(bounds)
    ax2.set_ylim(bounds)
    ax2.set_aspect('equal', adjustable='box')
    plt.plot(bounds, bounds, alpha=0.5, color='black', ls='--', lw=2)
    plt.legend(["identity", label_test])