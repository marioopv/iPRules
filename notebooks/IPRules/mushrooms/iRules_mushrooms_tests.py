import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import Bunch

from iPRules.iPRules import iPRules
from notebooks.IPRules.test_utils import one_hot_encode_dataframe, generate_results

# Load Dataset

# Mushrooms
filename = 'mushrooms'
target_true = 'p'
target_false = 'e'
test_size = 0.2

data_file_name = f'../../data/{filename}.csv'
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

encoded_pandas_dataset, encoded_feature_names = one_hot_encode_dataframe(pandas_dataset, feature_names)

X = encoded_pandas_dataset[encoded_feature_names]
y = encoded_pandas_dataset[target_value_name]

encoded_dataset = Bunch(
    data=X.to_numpy(),
    target=y.to_numpy(),
    target_names=target_value_name,
    feature_names=X.columns
)

# Define scorer
ensemble = RandomForestClassifier()

# Different values
scale_feature_coefficient = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
min_accuracy_coefficient = [0.7, 0.8, 0.9]
min_number_class_per_node = [3, 5, 7]
# chi_square_percent_point_function = [0.95, 0.96, 0.97, 0.98, 0.99]  # Not sure about this one
chi_square_percent_point_function = [0.95]  # Not sure about this one
#sorting_method = ['target_accuracy']
sorting_method = ['target_accuracy', 'complexity', 'chi2_statistic']

generate_results(ensemble,
                 filename, encoded_dataset, test_size,
                 chi_square_percent_point_function,
                 scale_feature_coefficient,
                 min_accuracy_coefficient,
                 min_number_class_per_node,
                 sorting_method)
