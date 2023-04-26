import pandas as pd
from sklearn.utils import Bunch

from notebooks.IPRules.test_utils import generate_battery_test, one_hot_encode_dataframe, continuous_to_discrete_column

# Load Dataset
filename = 'segmentation'
target_true = 'BRICKFACE'
target_false = 'REMAINDER'

list_of_continuous_columns = ["REGION-CENTROID-COL", "REGION-CENTROID-ROW", "VEDGE-MEAN", "VEDGE-SD", "HEDGE-MEAN",
                              "HEDGE-SD", "INTENSITY-MEAN", "RAWRED-MEAN", "RAWBLUE-MEAN", "RAWGREEN-MEAN",
                              "EXRED-MEAN", "EXBLUE-MEAN", "EXGREEN-MEAN", "VALUE-MEAN", "SATURATION-MEAN",
                              "HUE-MEAN"]
list_of_continuous_columns = [sub.replace('-', '_').replace(' ', '') for sub in list_of_continuous_columns]
number_of_divisions = 5

data_file_name = f'../../../../data/{filename}.csv'
results_file_name = f'../../../../Results/battery_test_{filename}.csv'

pandas_dataset = pd.read_csv(data_file_name)
pandas_dataset = pandas_dataset.replace('?', 'unknown')
pandas_dataset = pandas_dataset.dropna()
pandas_dataset.columns = [sub.replace('%', '') for sub in pandas_dataset.columns]
target_value_name = pandas_dataset.columns[-1]
pandas_dataset[target_value_name] = pandas_dataset[target_value_name].map({target_false: 0, target_true: 1})
pandas_dataset.columns = [sub.replace('-', '_').replace(' ', '').replace('class', 'target_value') for sub in
                          pandas_dataset.columns]
target_value_name = pandas_dataset.columns[-1]
feature_names = pandas_dataset.columns[0:-1]

pandas_dataset = continuous_to_discrete_column(pandas_dataset, list_of_continuous_columns, number_of_divisions)
encoded_pandas_dataset, encoded_feature_names = one_hot_encode_dataframe(pandas_dataset, feature_names)

X = encoded_pandas_dataset[encoded_feature_names]
y = encoded_pandas_dataset[target_value_name]

encoded_dataset = Bunch(
    data=X.to_numpy(),
    target=y.to_numpy(),
    target_names=target_value_name,
    feature_names=X.columns
)

# Different values
n_splits = 10
chi_square_percent_point_function = 0.95
scale_feature_coefficient = 0.01
min_accuracy_coefficient = 0.9
min_number_class_per_node = 3
sorting_method = "target_accuracy"

f_score = generate_battery_test(filename, results_file_name,X, y, encoded_dataset, target_value_name, n_splits, chi_square_percent_point_function,
                          scale_feature_coefficient, min_accuracy_coefficient, min_number_class_per_node,
                          sorting_method)

