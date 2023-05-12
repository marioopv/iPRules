from notebooks.IPRules.read_datasets import read_dataset
from notebooks.IPRules.test_utils import generate_results

# Load Dataset
# filename = "ionosphere"
# filename = "salud-covid"
# filename = "kr-vs-kp"
# filename = "divorce"
# filename = "connect-4"
# filename = "SPECT"
filename = "credit"

path = f'../../..'

results_file_name = f'{path}/Results/battery_{filename}.csv'
dataset_path_name = f'{path}/data/{filename}.csv'
test_size = 0.2
X, y, dataset, target_value_name, pandas_dataset = read_dataset(filename, dataset_path_name)


n_splits = 10
n_repeats = 3
# Different values
# TIME CONSUMING
# criterion = ["gini", "entropy", "log_loss"]
criterion = ["gini"]
scale_feature_coefficient = [0.01, 0.05, 0.1, 0.2, 0.2]
min_number_class_per_node = [3, 5, 7, 10]

# NOT TIME CONSUMING
min_accuracy_coefficient = [0.85, 0.9, 0.95]
chi_square_percent_point_function = [0.95, 0.97, 0.98, 0.99, 0.995]
sorting_method = ['target_accuracy']

print(filename)
generate_results(results_file_name, X, y, dataset, test_size,
                 chi_square_percent_point_function,
                 scale_feature_coefficient,
                 min_accuracy_coefficient,
                 min_number_class_per_node,
                 sorting_method, criterion, n_splits, n_repeats)
