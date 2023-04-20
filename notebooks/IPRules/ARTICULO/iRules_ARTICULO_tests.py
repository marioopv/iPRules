import pandas as pd
from sklearn.utils import Bunch
from notebooks.IPRules.test_utils import generate_results

# Load Dataset
filename = 'clean_dataset'
test_size = 0.2

data_file_name = f'../../data/{filename}.csv'
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
scale_feature_coefficient = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
min_accuracy_coefficient = [0.66, 0.8, 0.85, 0.9, 0.95]
min_number_class_per_node = [2, 3, 5, 7, 9, 11]
chi_square_percent_point_function = [0.95, 0.97, 0.99, 0.995]
sorting_method = ['target_accuracy', 'complexity', 'chi2_statistic']

generate_results(filename, dataset, test_size,
                 chi_square_percent_point_function,
                 scale_feature_coefficient,
                 min_accuracy_coefficient,
                 min_number_class_per_node,
                 sorting_method)
