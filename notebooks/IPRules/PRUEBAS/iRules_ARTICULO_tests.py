import pandas as pd
from sklearn.utils import Bunch
from notebooks.IPRules.test_utils import generate_results

# Load Dataset
filename = 'clean_dataset'
test_size = 0.2

data_file_name = f'../../data/{filename}.csv'
results_file_name = f'../../Results/Results_{filename}.csv'


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
# TIME CONSUMING
criterion = ["gini", "entropy", "log_loss"]
scale_feature_coefficient = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
min_number_class_per_node = [1, 2, 3, 5]

# NOT TIME CONSUMING
min_accuracy_coefficient = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
chi_square_percent_point_function = [0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
sorting_method = ['target_accuracy', 'complexity', 'chi2_statistic']

generate_results(results_file_name, dataset, test_size,
                 chi_square_percent_point_function,
                 scale_feature_coefficient,
                 min_accuracy_coefficient,
                 min_number_class_per_node,
                 sorting_method, criterion)
