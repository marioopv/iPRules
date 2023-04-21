import numpy as np
import pandas as pd
import copy
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


def generate_results(results_file_name, dataset, test_size,
                     chi_square_percent_point_function,
                     scale_feature_coefficient,
                     min_accuracy_coefficient,
                     min_number_class_per_node,
                     sorting_method, criterion="gini"):
    # WRITE FILE
    f = open(results_file_name, "w")
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
        generate_results_from_criterion(X_test, X_train, chi_square_percent_point_function, criteria, dataset, f,
                                        min_accuracy_coefficient, min_number_class_per_node, scale_feature_coefficient,
                                        sorting_method, train_pandas_dataset, y_pred_test_RuleFit, y_pred_test_tree,
                                        y_test, y_train)
    f.close()


def generate_results_from_criterion(X_test, X_train, chi_square_percent_point_function, criteria, dataset, f,
                                    min_accuracy_coefficient, min_number_class_per_node, scale_feature_coefficient,
                                    sorting_method, train_pandas_dataset, y_pred_test_RuleFit, y_pred_test_tree, y_test,
                                    y_train):
    # ENSEMBLE
    ensemble = RandomForestClassifier(criterion=criteria)
    ensemble.fit(X_train, y_train)
    y_pred_test_ensemble = ensemble.predict(X_test)
    for scaler in scale_feature_coefficient:
        for min_class in min_number_class_per_node:
            # PRECALCULATE DICT IN ORDER TO AVOID RERUNS OF A TIME CONSUMING PART OF THE CODE
            rules = iPRules(
                feature_names=dataset.feature_names,
                target_value_name=dataset.target_names,
                scale_feature_coefficient=scaler,
                min_number_class_per_node=min_class
            )
            dict_nodes, most_important_features = rules.generate_nodes(train_pandas_dataset, ensemble.feature_importances_)
            for min_accuracy in min_accuracy_coefficient:
                for chi2 in chi_square_percent_point_function:
                    new_rules = iPRules(
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
                    calculated = new_rules.fit(
                        pandas_dataset=train_pandas_dataset,
                        feature_importances=ensemble.feature_importances_,
                        node_dict=copy.deepcopy(dict_nodes),
                        most_important_features=copy.deepcopy(most_important_features)
                    )

                    if not calculated:
                        print("NOT CALCULATED")
                        continue

                    print_results(X_test, chi2, criteria, f, min_accuracy, min_class, new_rules, rules, scaler,
                                  sorting_method, y_pred_test_RuleFit, y_pred_test_ensemble, y_pred_test_tree,
                                  y_test)


def print_results(X_test, chi2, criteria, f, min_accuracy, min_class, new_rules, rules, scaler, sorting_method,
                  y_pred_test_RuleFit, y_pred_test_ensemble, y_pred_test_tree, y_test):
    for sorting in sorting_method:
        if not new_rules.rules_:
            empty_restuls(chi2, criteria, f, min_accuracy, min_class, scaler, y_test)
            continue
        # RULES
        y_pred_test_rules = new_rules.predict(X_test, sorting_method=sorting)

        if len(y_pred_test_rules) == 0:
            print("0 MATHS IN TEST")
            empty_restuls(chi2, criteria, f, min_accuracy, min_class, scaler, y_test)
            continue
        line_results = generate_line_results(chi2, criteria, min_accuracy, min_class, rules, scaler, sorting,
                                             y_pred_test_RuleFit, y_pred_test_ensemble, y_pred_test_rules,
                                             y_pred_test_tree, y_test)
        print(line_results)
        f.write(line_results)


def empty_restuls(chi2, criteria, f, min_accuracy, min_class, scaler, y_test):
    print('empty list no rules')
    tttt = f'{criteria}, {chi2}, {scaler}, {min_accuracy}, {min_class}, NaN, {len(y_test)}, NaN, 0' \
           f', NaN, NaN, NaN, NaN, NaN' \
           f', NaN, NaN, NaN, NaN, NaN' \
           f', NaN, NaN, NaN, NaN, NaN' \
           f', NaN, NaN, NaN, NaN, NaN\n'
    f.write(tttt)
    print(tttt)


def generate_line_results(chi2, criteria, min_accuracy, min_class, rules, scaler, sorting, y_pred_test_RuleFit,
                y_pred_test_ensemble, y_pred_test_rules, y_pred_test_tree, y_test):
    # TODO: DIVIDE METHODS
    # DATASET CATEGORIZABLES
    np_array_rules = np.array(y_pred_test_rules)
    filter_indices = np.where(np_array_rules != None)[0]
    filtered_y_test = np.array(y_test)[filter_indices].astype('int64')
    filtered_y_pred_test_ensemble = np.array(y_pred_test_ensemble)[filter_indices].astype('int64')
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
    ensemble_precision_score = metrics.precision_score(filtered_y_test,filtered_y_pred_test_ensemble)
    tree_precision_score = metrics.precision_score(filtered_y_test, filtered_y_pred_test_tree)
    RuleFit_precision_score = metrics.precision_score(filtered_y_test,filtered_y_pred_test_RuleFit)
    rules_precision_score = metrics.precision_score(filtered_y_test, filtered_y_pred_test_rules)
    # Recall
    ensemble_recall = metrics.recall_score(filtered_y_test, filtered_y_pred_test_ensemble)
    tree_recall_score = metrics.recall_score(filtered_y_test, filtered_y_pred_test_tree)
    RuleFit_recall_score = metrics.recall_score(filtered_y_test, filtered_y_pred_test_RuleFit)
    rules_recall_score = metrics.recall_score(filtered_y_test, filtered_y_pred_test_rules)

    # ROC AUC
    ensemble_roc_auc_score = metrics.roc_auc_score(filtered_y_test,filtered_y_pred_test_ensemble)
    tree_roc_auc_score = metrics.roc_auc_score(filtered_y_test, filtered_y_pred_test_tree)
    RuleFit_roc_auc_score = metrics.roc_auc_score(filtered_y_test, filtered_y_pred_test_RuleFit)
    rules_roc_auc_score = metrics.roc_auc_score(filtered_y_test, filtered_y_pred_test_rules)

    line_results = f'{criteria}, {chi2}, {scaler}, {min_accuracy}, {min_class}, {sorting}, {len(y_test)}, {len(filtered_y_test)}, {len(rules.rules_)}, {len(filtered_y_pred_test_rules) / len(y_test)}'
    line_results += f', {ensemble_accuracy}, {ensemble_f1_score}, {ensemble_precision_score}, {ensemble_recall}, {ensemble_roc_auc_score}'
    line_results += f', {tree_accuracy}, {tree_f1_score}, {tree_precision_score}, {tree_recall_score}, {tree_roc_auc_score}'
    line_results += f', {RuleFit_accuracy}, {RuleFit_f1_score}, {RuleFit_precision_score}, {RuleFit_recall_score}, {RuleFit_roc_auc_score}'
    line_results += f', {rules_accuracy}, {rules_f1_score}, {rules_precision_score}, {rules_recall_score}, {rules_roc_auc_score}\n'
    return line_results
