import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from iPRules.iPRules import iPRules


def one_hot_encode_dataframe(data, feature_names):
    enc = OneHotEncoder(sparse_output=False)
    encoded_array = enc.fit_transform(data.loc[:, feature_names])
    encoded_feature_names = enc.get_feature_names_out()
    df_encoded = pd.DataFrame(encoded_array, columns=encoded_feature_names)
    encoded_pandas_dataset = pd.concat([df_encoded, data], axis=1)
    encoded_pandas_dataset.drop(labels=feature_names, axis=1, inplace=True)
    return encoded_pandas_dataset, encoded_feature_names


def generate_results(ensemble,
                     filename, dataset, test_size,
                     chi_square_percent_point_function,
                     scale_feature_coefficient,
                     min_accuracy_coefficient,
                     min_number_class_per_node,
                     sorting_method):
    # Define dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=test_size,
                                                        random_state=1)
    train_pandas_dataset = pd.DataFrame(data=np.c_[X_train, y_train],
                                        columns=list(dataset['feature_names']) + [dataset.target_names])

    print('Sizes (without target):')
    print(f'Original size {dataset.data.shape}')
    print(f'Train size {X_train.shape}')
    print(f'Test size {X_test.shape}')

    # ENSEMBLE
    ensemble.fit(X_train, y_train)
    y_pred_test_ensemble = ensemble.predict(X_test)

    data_file_name = f'../../data/Test_{filename}.txt'
    f = open(data_file_name, "a")
    file_header = "chi_square_percent_point_function, scale_feature_coefficient, min_accuracy_coefficient, " \
                  "min_number_class_per_node, sorting_method, dataset_size, dataset_categorizable, number_of_rules, " \
                  "cobertura, RF_accuracy, RULES_accuracy\n"
    print(file_header)
    f.write(file_header)

    for scaler in scale_feature_coefficient:
        for min_accuracy in min_accuracy_coefficient:
            for min_class in min_number_class_per_node:
                for chi2 in chi_square_percent_point_function:
                    #print(f'chi2:{chi2}, scaler:{scaler}, min_accuracy:{min_accuracy}, min_class:{min_class}')

                    rules = iPRules(
                        # display_logs=True,
                        feature_names=dataset.feature_names,
                        target_value_name=dataset.target_names,
                        chi_square_percent_point_function=chi2,  # como si fuese la inversa del p-value
                        scale_feature_coefficient=scaler,  # mas o menos variables
                        min_accuracy_coefficient=min_accuracy,
                        # precisión para definir si la regla es positiva o  negativa
                        min_number_class_per_node=min_class
                        # al menos cuantos registros tiene que haber para que se pueda definir si es regla
                    )
                    # Fit model
                    rules.fit(train_pandas_dataset, ensemble.feature_importances_)

                    if not rules.rules_:
                        print('empty list no rules')
                        tttt = f'{chi2}, {scaler}, {min_accuracy}, {min_class}, {sorting}, {len(y_test)}, NaN, 0, ' \
                               f'NaN, NaN, NaN\n'
                        f.write(tttt)
                        print(tttt)
                        continue

                    for sorting in sorting_method:
                        # RULES
                        y_pred_test_rules = rules.predict(X_test, sorting_method=sorting)

                        # CATEGORIZABLES
                        np_array_rules = np.array(y_pred_test_rules)
                        # not_filter_indices = np.where(np.logical_and(np_array_rules != 0, np_array_rules!=1))[0]
                        filter_indices = np.where(np_array_rules != None)[0]

                        np_filterred_y_test = np.array(y_test)[filter_indices].astype('int64')
                        np_filterred_y_pred_test_ensemble = np.array(y_pred_test_ensemble)[filter_indices].astype(
                            'int64')
                        np_filterred_y_pred_test_rules = np.array(y_pred_test_rules)[filter_indices].astype('int64')

                        # ACCURACY
                        ensemble_accuracy = metrics.accuracy_score(np_filterred_y_test,
                                                                   np_filterred_y_pred_test_ensemble,
                                                                   normalize=True)
                        rules_accuracy = metrics.accuracy_score(np_filterred_y_test,
                                                                np_filterred_y_pred_test_rules,
                                                                normalize=True)
                        tttt = f'{chi2}, {scaler}, {min_accuracy}, {min_class}, {sorting}, {len(y_test)}, {len(np_filterred_y_test)}, {len(rules.rules_)}, {len(np_filterred_y_pred_test_rules) / len(y_test)}, {ensemble_accuracy}, {rules_accuracy}\n'
                        print(tttt)
                        f.write(tttt)
    f.close()
