from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from utils import feature_importance_based_on_mean_decrease_in_impurity, print_forest_trees, \
    feature_importance_based_on_feature_permutation, print_results
from sklearn.model_selection import train_test_split


# Load the Iris dataset
dataset = load_breast_cancer()

# Split features and target into train and test sets
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.33, random_state=1)

# Create a random forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=5, random_state=42)

# Train the random forest classifier on the Iris dataset
forest.fit(X_train, y_train)
# Make predictions for the test set
y_pred_test = forest.predict(X_test)


# View the classification report for test data and predictions

print_results(forest, y_test, y_pred_test, X_test, plot_confusion_matrix=True)


# Feature importance based on mean decrease in impurity
feature_importance_based_on_mean_decrease_in_impurity(forest, dataset, plot_importances=True)
# Feature importance based on mean decrease in impurity
feature_importance_based_on_feature_permutation(forest, dataset, plot_importances=True)

# Display rules
print_forest_trees(forest, dataset, plot_trees=True)
