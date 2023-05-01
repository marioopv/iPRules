import pandas as pd
from sklearn.utils import Bunch

from notebooks.IPRules.test_utils import generate_battery_test

# Load Dataset
filename = 'clean_dataset'

data_file_name = f'../../../data/{filename}.csv'
results_file_name = f'../../../Results/Results_{filename}.csv'

pandas_dataset = pd.read_csv(data_file_name)
pandas_dataset = pandas_dataset.dropna()
target_value_name = pandas_dataset.columns[-1]
feature_names = pandas_dataset.columns[0:-1]
X = pandas_dataset[feature_names]
y = pandas_dataset[target_value_name]
dataset = Bunch(
    data=X.to_numpy(),
    target=y.to_numpy(),
    target_names=target_value_name,
    feature_names=X.columns
)

# Different values
n_splits = 10
chi_square_percent_point_function = 0.95
scale_feature_coefficient = 0.05
min_accuracy_coefficient = 0.9
min_number_class_per_node = 3
sorting_method = "target_accuracy"
criterion = "entropy"

np_cobertura_list, \
    np_RuleFit_accuracy_list, np_RuleFit_f1_score_list, np_RuleFit_precision_score_list, np_RuleFit_recall_list, np_RuleFit_roc_auc_score_list, \
    np_ensemble_accuracy_list, np_ensemble_f1_score_list, np_ensemble_precision_score_list, np_ensemble_recall_list, np_ensemble_roc_auc_score_list, \
    np_rules_accuracy_list, np_rules_f1_score_list, np_rules_precision_score_list, np_rules_recall_list, np_rules_roc_auc_score_list, \
    np_tree_accuracy_list, np_tree_f1_score_list, np_tree_precision_score_list, np_tree_recall_list, np_tree_roc_auc_score_list \
    = generate_battery_test(X, y, dataset, target_value_name, n_splits, chi_square_percent_point_function,
                            scale_feature_coefficient, min_accuracy_coefficient, min_number_class_per_node,
                            sorting_method, criterion)

print("%0.5f np_cobertura_list with a standard deviation of %0.5f" % (np_cobertura_list.mean(), np_cobertura_list.std()))
print("%0.5f np_RuleFit_accuracy_list with a standard deviation of %0.5f" % (np_RuleFit_accuracy_list.mean(), np_RuleFit_accuracy_list.std()))
print("%0.5f np_ensemble_accuracy_list with a standard deviation of %0.5f" % ( np_ensemble_accuracy_list.mean(), np_ensemble_accuracy_list.std()))
print("%0.5f np_rules_accuracy_list with a standard deviation of %0.5f" % ( np_rules_accuracy_list.mean(), np_rules_accuracy_list.std()))
print("%0.5f np_tree_accuracy_list with a standard deviation of %0.5f" % (np_tree_accuracy_list.mean(), np_tree_accuracy_list.std()))
print(np_cobertura_list, \
      np_RuleFit_accuracy_list, np_RuleFit_f1_score_list, np_RuleFit_precision_score_list, np_RuleFit_recall_list,
      np_RuleFit_roc_auc_score_list, \
      np_ensemble_accuracy_list, np_ensemble_f1_score_list, np_ensemble_precision_score_list, np_ensemble_recall_list,
      np_ensemble_roc_auc_score_list, \
      np_rules_accuracy_list, np_rules_f1_score_list, np_rules_precision_score_list, np_rules_recall_list,
      np_rules_roc_auc_score_list, \
      np_tree_accuracy_list, np_tree_f1_score_list, np_tree_precision_score_list, np_tree_recall_list,
      np_tree_roc_auc_score_list)