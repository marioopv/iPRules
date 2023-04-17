import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import Bunch

from iPRules.iPRules import iPRules


def one_hot_encode_dataframe(data, feature_names):
    enc = OneHotEncoder(sparse_output=False)
    encoded_array = enc.fit_transform(data.loc[:, feature_names])
    encoded_feature_names = enc.get_feature_names_out()
    df_encoded = pd.DataFrame(encoded_array, columns=encoded_feature_names)
    encoded_pandas_dataset = pd.concat([df_encoded, data], axis=1)
    encoded_pandas_dataset.drop(labels=feature_names, axis=1, inplace=True)
    return encoded_pandas_dataset, encoded_feature_names


# Load Dataset
# iris
# dataset = load_iris()
target_value_name = 'class'

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

# dataset.feature_names = [sub.replace(' ', '').replace('(cm)', '') for sub in dataset.feature_names]

pandas_dataset.head()

encoded_pandas_dataset, encoded_feature_names = one_hot_encode_dataframe(pandas_dataset, feature_names)
encoded_pandas_dataset.head()

X = encoded_pandas_dataset[encoded_feature_names]
y = encoded_pandas_dataset[target_value_name]

encoded_dataset = Bunch(
    data=X.to_numpy(),
    target=y.to_numpy(),
    target_names=target_value_name,
    feature_names=X.columns
)

# Define dataset
X_train, X_test, y_train, y_test = train_test_split(encoded_dataset.data, encoded_dataset.target, test_size=test_size,
                                                    random_state=1)
encoded_train_pandas_dataset = pd.DataFrame(data=np.c_[X_train, y_train],
                                            columns=list(encoded_dataset['feature_names']) + [target_value_name])
encoded_test_pandas_dataset = pd.DataFrame(data=np.c_[X_test, y_test],
                                           columns=list(encoded_dataset['feature_names']) + [target_value_name])
print()
print('Sizes (without target):')
print(f'Original size {encoded_dataset.data.shape}')
print(f'Train size {X_train.shape}')
print(f'Test size {X_test.shape}')

# Define scorer
ensemble = RandomForestClassifier(n_estimators=100)
ensemble.fit(X_train, y_train)

# ENSEMBLE
y_pred_test_ensemble = ensemble.predict(X_test)

chi_square_percent_point_function = [0.95, 0.96, 0.97],
scale_feature_coefficient = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2],
min_accuracy_coefficient = [0.7, 0.8, 0.9],
min_number_class_per_node = [3, 5, 7, 9, 12],
sorting_method = ['target_accuracy', 'complexity', 'p_value', 'chi2_statistic']

data_file_name = f'../../data/Test_{filename}.txt'

f = open(data_file_name, "w")

f.write('chi_square_percent_point_function, scale_feature_coefficient, min_accuracy_coefficient, '
        'min_number_class_per_node, dataset_size, dataset_categorizable, cobertura, RF_accuracy, RULES_accuracy')

for chi2 in chi_square_percent_point_function:
    for scaler in scale_feature_coefficient:
        for min_accuracy in min_accuracy_coefficient:
            for min_class in min_number_class_per_node:
                rules = iPRules(
                    base_ensemble=ensemble,
                    feature_names=encoded_dataset.feature_names,
                    target_value_name=encoded_dataset.target_names,
                    chi_square_percent_point_function=0.95,  # como si fuese la inversa del p-value
                    scale_feature_coefficient=0.01,  # mas o menos variablses
                    min_accuracy_coefficient=0.95,  # precisión para definir si la regla es positiva o  negativa
                    min_number_class_per_node=3
                    # al menos cuantos registros tiene que haber para que se pueda definir si es regla
                )
                # Fit model
                rules.fit(encoded_train_pandas_dataset)

                for sorting in sorting_method:
                    # RULES
                    y_pred_test_rules = rules.predict(X_test, sorting_method="target_accuracy")

                    # CATEGORIZABLES
                    np_array_rules = np.array(y_pred_test_rules)
                    # not_filter_indices = np.where(np.logical_and(np_array_rules != 0, np_array_rules!=1))[0]
                    filter_indices = np.where(np_array_rules != None)[0]

                    np_filterred_y_test = np.array(y_test)[filter_indices].astype('int64')
                    np_filterred_y_pred_test_ensemble = np.array(y_pred_test_ensemble)[filter_indices].astype('int64')
                    np_filterred_y_pred_test_rules = np.array(y_pred_test_rules)[filter_indices].astype('int64')
                    # CHANGE FORMAT IN ORDER TO NOT HAVE PROBLEMS
                    np_filterred_y_pred_test_rules = np_filterred_y_pred_test_rules.astype('int64')
                    ensemble_accuracy = metrics.accuracy_score(np_filterred_y_test, np_filterred_y_pred_test_ensemble,
                                                               normalize=True)
                    rules_accuracy = metrics.accuracy_score(np_filterred_y_test, np_filterred_y_pred_test_rules,
                                                            normalize=True)
                    text = f'{chi2}, {scaler}, {min_accuracy}, {min_class}, {len(y_test)}, {len(np_filterred_y_test)}, {len(np_filterred_y_pred_test_rules) / len(y_test)}, {ensemble_accuracy}, {rules_accuracy}'
                    f.write(text)

f.close()
