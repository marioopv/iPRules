import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import export_text, plot_tree, export_graphviz
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import train_test_split


def feature_importance_based_on_mean_decrease_in_impurity(forest, dataset, plot_importances=True):
    """
    Feature importance based on mean decrease in impurity¶
    Feature importances are provided by the fitted attribute feature_importances_ and they are computed as the mean and standard deviation of accumulation of the impurity decrease within each tree.

Warning Impurity-based feature importances can be misleading for high cardinality features (many unique values). See Permutation feature importance as an alternative below.
    :param plot_importances:
    :param forest:
    :param dataset:
    :return:
    """
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    forest_importances = pd.Series(importances, index=dataset.feature_names)

    if plot_importances:
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show()
    # print(f'"Feature importances using MDI {forest_importances}')
    return forest_importances


def feature_importance_based_on_feature_permutation(forest, dataset, plot_importances=True):
    """
    Feature importance based on feature permutation¶

    Permutation feature importance overcomes limitations of the impurity-based feature importance: they do not have a bias toward high-cardinality features and can be computed on a left-out test set.
    :param print_importances:
    :param forest:
    :param dataset:
    :return:
    """
    result = permutation_importance(
        forest, dataset.data, dataset.target, n_repeats=10, random_state=42, n_jobs=2
    )
    forest_importances = pd.Series(result.importances_mean, index=dataset.feature_names)

    if plot_importances:
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
        ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        plt.show()
    # print(f'"Feature importances using permutation on full model {forest_importances}')

    return forest_importances


def print_forest_trees(forest, dataset, plot_trees=False):
    """
     Loop over each tree in the forest and export its text representation
    :param plot_trees:
    :param forest:
    :param dataset:
    :return:
    """
    for i, tree in enumerate(forest.estimators_):
        r = export_text(tree, feature_names=list(dataset.feature_names))
        print(f'Tree {i}:\n{r}\n')
        if plot_trees:
            # Creating the tree plot
            plot_tree(tree, filled=True)
            plt.show()
        # TODO: save and print in latex
        # print(export_graphviz(tree))


def print_results(forest, y_test, y_pred_test, X_test, plot_confusion_matrix=True):
    """
    View the classification report for test data and predictions
    :param forest:
    :param y_test:
    :param y_pred_test:
    :return:
    """
    print(classification_report(y_test, y_pred_test, digits=4))
    if plot_confusion_matrix:
        confusion_matriz = confusion_matrix(y_test, y_pred_test, labels=forest.classes_)
        confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matriz, display_labels=forest.classes_)

        roc_display = RocCurveDisplay.from_estimator(forest, X_test, y_test)
        pr_display = PrecisionRecallDisplay.from_estimator(forest, X_test, y_test)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))
        roc_display.plot(ax=ax1)
        pr_display.plot(ax=ax2)
        confusion_matrix_display.plot(ax=ax3)
        plt.show()
