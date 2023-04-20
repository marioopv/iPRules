import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from iPRules.iPRules import iPRules
from imodels import RuleFitClassifier


def one_hot_encode_dataframe(data, feature_names):
    enc = OneHotEncoder(sparse_output=False)
    encoded_array = enc.fit_transform(data.loc[:, feature_names])
    encoded_feature_names = enc.get_feature_names_out()
    df_encoded = pd.DataFrame(encoded_array, columns=encoded_feature_names)
    encoded_pandas_dataset = pd.concat([df_encoded, data], axis=1)
    encoded_pandas_dataset.drop(labels=feature_names, axis=1, inplace=True)
    return encoded_pandas_dataset, encoded_feature_names


def generate_results(filename, dataset, test_size,
                     chi_square_percent_point_function,
                     scale_feature_coefficient,
                     min_accuracy_coefficient,
                     min_number_class_per_node,
                     sorting_method, criterion="gini"):
    # WRITE FILE

    data_file_name = f'../../../Results/Results_{filename}.csv'
    f = open(data_file_name, "w")
    file_header = "ensemble_criterion, chi_square_percent_point_function, scale_feature_coefficient, min_accuracy_coefficient, " \
                  "min_number_class_per_node, sorting_method" \
                  ", dataset_test_size, dataset_test_categorizable" \
                  ", number_of_rules, cobertura"
    file_header += ', ensemble_accuracy, ensemble_f1_score, ensemble_precision_score, ensemble_recall, ensemble_roc_auc_score'
    file_header += ', tree_accuracy, tree_f1_score, tree_precision_score, tree_recall_score, tree_roc_auc_score'
    file_header += ', RuleFit_accuracy, RuleFit_f1_score, RuleFit_precision_score, RuleFit_recall_score, RuleFit_roc_auc_score'
    file_header += ', rules_accuracy, rules_f1_score, rules_precision_score, rules_recall_score, rules_roc_auc_score\n'

    print(file_header)
    f.write(file_header)

    # Define dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=test_size,
                                                        random_state=1)
    train_pandas_dataset = pd.DataFrame(data=np.c_[X_train, y_train],
                                        columns=list(dataset['feature_names']) + [dataset.target_names])

    print('Sizes (without target):')
    print(f'Original size {dataset.data.shape}')
    print(f'Train size {X_train.shape}')
    print(f'Test size {X_test.shape}')

    # TREE
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)
    y_pred_test_tree = tree.predict(X_test)

    # RuleFit
    tree = RuleFitClassifier()
    tree.fit(X_train, y_train, feature_names=dataset.feature_names)
    y_pred_test_RuleFit = tree.predict(X_test)

    for criteria in criterion:

        # ENSEMBLE
        ensemble = RandomForestClassifier(criterion=criteria)
        ensemble.fit(X_train, y_train)
        y_pred_test_ensemble = ensemble.predict(X_test)

        for scaler in scale_feature_coefficient:
            for min_accuracy in min_accuracy_coefficient:
                for min_class in min_number_class_per_node:
                    for chi2 in chi_square_percent_point_function:

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
                            tttt = f'{criteria}, {chi2}, {scaler}, {min_accuracy}, {min_class}, {sorting}, {len(y_test)}, NaN, 0' \
                                   f', NaN, NaN, NaN, NaN, NaN' \
                                   f', NaN, NaN, NaN, NaN, NaN' \
                                   f', NaN, NaN, NaN, NaN, NaN' \
                                   f', NaN, NaN, NaN, NaN, NaN\n'
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

                            filtered_y_test = np.array(y_test)[filter_indices].astype('int64')
                            filtered_y_pred_test_ensemble = np.array(y_pred_test_ensemble)[filter_indices].astype(
                                'int64')
                            filtered_y_pred_test_tree = np.array(y_pred_test_tree)[filter_indices].astype('int64')
                            filtered_y_pred_test_RuleFit = np.array(y_pred_test_RuleFit)[filter_indices].astype('int64')
                            filtered_y_pred_test_rules = np.array(y_pred_test_rules)[filter_indices].astype('int64')

                            # ACCURACY
                            ensemble_accuracy = metrics.accuracy_score(filtered_y_test, filtered_y_pred_test_ensemble)
                            tree_accuracy = metrics.accuracy_score(filtered_y_test, filtered_y_pred_test_tree)
                            RuleFit_accuracy = metrics.accuracy_score(filtered_y_test, filtered_y_pred_test_RuleFit)
                            rules_accuracy = metrics.accuracy_score(filtered_y_test, filtered_y_pred_test_rules)

                            # F1
                            ensemble_f1_score = metrics.f1_score(filtered_y_test, filtered_y_pred_test_ensemble)
                            tree_f1_score = metrics.f1_score(filtered_y_test, filtered_y_pred_test_tree)
                            RuleFit_f1_score = metrics.f1_score(filtered_y_test, filtered_y_pred_test_RuleFit)
                            rules_f1_score = metrics.f1_score(filtered_y_test, filtered_y_pred_test_rules)

                            # Precision
                            ensemble_precision_score = metrics.precision_score(filtered_y_test,
                                                                               filtered_y_pred_test_ensemble)
                            tree_precision_score = metrics.precision_score(filtered_y_test, filtered_y_pred_test_tree)
                            RuleFit_precision_score = metrics.precision_score(filtered_y_test,
                                                                              filtered_y_pred_test_RuleFit)
                            rules_precision_score = metrics.precision_score(filtered_y_test, filtered_y_pred_test_rules)

                            # Recall
                            ensemble_recall = metrics.recall_score(filtered_y_test, filtered_y_pred_test_ensemble)
                            tree_recall_score = metrics.recall_score(filtered_y_test, filtered_y_pred_test_tree)
                            RuleFit_recall_score = metrics.recall_score(filtered_y_test, filtered_y_pred_test_RuleFit)
                            rules_recall_score = metrics.recall_score(filtered_y_test, filtered_y_pred_test_rules)
                            # ROC AUC
                            ensemble_roc_auc_score = metrics.roc_auc_score(filtered_y_test,
                                                                           filtered_y_pred_test_ensemble)
                            tree_roc_auc_score = metrics.roc_auc_score(filtered_y_test, filtered_y_pred_test_tree)
                            RuleFit_roc_auc_score = metrics.roc_auc_score(filtered_y_test, filtered_y_pred_test_RuleFit)
                            rules_roc_auc_score = metrics.roc_auc_score(filtered_y_test, filtered_y_pred_test_rules)

                            tttt = f'{criteria}, {chi2}, {scaler}, {min_accuracy}, {min_class}, {sorting}, {len(y_test)}, {len(filtered_y_test)}, {len(rules.rules_)}, {len(filtered_y_pred_test_rules) / len(y_test)}'
                            tttt += f', {ensemble_accuracy}, {ensemble_f1_score}, {ensemble_precision_score}, {ensemble_recall}, {ensemble_roc_auc_score}'
                            tttt += f', {tree_accuracy}, {tree_f1_score}, {tree_precision_score}, {tree_recall_score}, {tree_roc_auc_score}'
                            tttt += f', {RuleFit_accuracy}, {RuleFit_f1_score}, {RuleFit_precision_score}, {RuleFit_recall_score}, {RuleFit_roc_auc_score}'
                            tttt += f', {rules_accuracy}, {rules_f1_score}, {rules_precision_score}, {rules_recall_score}, {rules_roc_auc_score}\n'
                            print(tttt)
                            f.write(tttt)
    f.close()
