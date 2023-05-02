import copy

import numpy as np
import pandas as pd
from imodels import RuleFitClassifier
from rulecosi import RuleCOSIClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import dask.dataframe as dd

from iPRules.iPRules import iPRules


def continuous_to_discrete_column(dataset, list_of_continuous_columns, number_of_divisions=10):
    labels = []

    if number_of_divisions == 10:
        labels = ['L_VeryLow', 'L_Low', 'L_Medium', 'L_High', 'L_VeryHigh',
                  'R_VeryLow', 'R_Low', 'R_Medium', 'R_High', 'R_VeryHigh']
    if number_of_divisions == 5:
        labels = ['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh']

    for column_name in list_of_continuous_columns:
        dataset[column_name] = pd.cut(
            dataset[column_name]
            , number_of_divisions
            , labels=labels
        )

    return dataset


def clean_names(name_list):
    return [sub.replace('?', '_').replace('.', '_').replace('-', '_').replace(' ', '') for sub in name_list]


def one_hot_encode_dataframe(data, feature_names):
    enc = OneHotEncoder(sparse_output=False)
    encoded_array = enc.fit_transform(data.loc[:, feature_names])
    encoded_feature_names = clean_names(enc.get_feature_names_out())
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
    ruleFit = RuleFitClassifier()
    ruleFit.fit(X_train, y_train, feature_names=dataset.feature_names)
    y_pred_test_RuleFit = ruleFit.predict(X_test)

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
            dict_nodes, most_important_features = rules.generate_nodes(train_pandas_dataset,
                                                                       ensemble.feature_importances_)
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
                        dataset=train_pandas_dataset,
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
    ensemble_precision_score = metrics.precision_score(filtered_y_test, filtered_y_pred_test_ensemble)
    tree_precision_score = metrics.precision_score(filtered_y_test, filtered_y_pred_test_tree)
    RuleFit_precision_score = metrics.precision_score(filtered_y_test, filtered_y_pred_test_RuleFit)
    rules_precision_score = metrics.precision_score(filtered_y_test, filtered_y_pred_test_rules)
    # Recall
    ensemble_recall = metrics.recall_score(filtered_y_test, filtered_y_pred_test_ensemble)
    tree_recall_score = metrics.recall_score(filtered_y_test, filtered_y_pred_test_tree)
    RuleFit_recall_score = metrics.recall_score(filtered_y_test, filtered_y_pred_test_RuleFit)
    rules_recall_score = metrics.recall_score(filtered_y_test, filtered_y_pred_test_rules)

    # ROC AUC
    ensemble_roc_auc_score = metrics.roc_auc_score(filtered_y_test, filtered_y_pred_test_ensemble)
    tree_roc_auc_score = metrics.roc_auc_score(filtered_y_test, filtered_y_pred_test_tree)
    RuleFit_roc_auc_score = metrics.roc_auc_score(filtered_y_test, filtered_y_pred_test_RuleFit)
    rules_roc_auc_score = metrics.roc_auc_score(filtered_y_test, filtered_y_pred_test_rules)

    line_results = f'{criteria}, {chi2}, {scaler}, {min_accuracy}, {min_class}, {sorting}, {len(y_test)}, {len(filtered_y_test)}, {len(rules.rules_)}, {len(filtered_y_pred_test_rules) / len(y_test)}'
    line_results += f', {ensemble_accuracy}, {ensemble_f1_score}, {ensemble_precision_score}, {ensemble_recall}, {ensemble_roc_auc_score}'
    line_results += f', {tree_accuracy}, {tree_f1_score}, {tree_precision_score}, {tree_recall_score}, {tree_roc_auc_score}'
    line_results += f', {RuleFit_accuracy}, {RuleFit_f1_score}, {RuleFit_precision_score}, {RuleFit_recall_score}, {RuleFit_roc_auc_score}'
    line_results += f', {rules_accuracy}, {rules_f1_score}, {rules_precision_score}, {rules_recall_score}, {rules_roc_auc_score}\n'
    return line_results


def generate_scores(filtered_y_test, filtered_y_pred_test_ensemble):
    accuracy = metrics.accuracy_score(filtered_y_test, filtered_y_pred_test_ensemble)
    f1_score = metrics.f1_score(filtered_y_test, filtered_y_pred_test_ensemble)
    precision_score = metrics.precision_score(filtered_y_test, filtered_y_pred_test_ensemble)
    recall = metrics.recall_score(filtered_y_test, filtered_y_pred_test_ensemble)

    try:
        roc_auc_score = metrics.roc_auc_score(filtered_y_test, filtered_y_pred_test_ensemble)
    except ValueError:
        roc_auc_score = 0.0

    return accuracy, f1_score, precision_score, recall, roc_auc_score


def generate_battery_test(f, filename, X, y, dataset, target_value_name, n_splits, n_repeats,
                          chi_square_percent_point_function,
                          scale_feature_coefficient, min_accuracy_coefficient, min_number_class_per_node,
                          sorting_method):
    cobertura_list, \
        RuleFit_accuracy_list, RuleFit_f1_score_list, RuleFit_precision_score_list, RuleFit_recall_list, RuleFit_roc_auc_score_list, \
        ensemble_accuracy_list, ensemble_f1_score_list, ensemble_precision_score_list, ensemble_recall_list, ensemble_roc_auc_score_list, \
        rules_accuracy_list, rules_f1_score_list, rules_precision_score_list, rules_recall_list, rules_roc_auc_score_list, \
        tree_accuracy_list, tree_f1_score_list, tree_precision_score_list, tree_recall_list, tree_roc_auc_score_list, \
        rulecosi_accuracy_list, rulecosi_f1_score_list, rulecosi_precision_score_list, rulecosi_recall_list, rulecosi_roc_auc_score_list, \
        rulefit_num_rules_list, rules_num_rules_list, rulecosi_num_rules_list \
        = kfold_test(X, chi_square_percent_point_function, dataset, min_accuracy_coefficient,
                     min_number_class_per_node, n_splits, n_repeats, scale_feature_coefficient, sorting_method,
                     target_value_name, y)

    f_score = f'{filename},F1-score,{round(cobertura_list.mean() * 100, 2)}±{round(cobertura_list.std() * 100, 2)}'
    f_score += f',{round(tree_f1_score_list.mean() * 100, 2)}±{round(tree_f1_score_list.std() * 100, 2)}'
    f_score += f',{round(ensemble_f1_score_list.mean() * 100, 2)}±{round(ensemble_f1_score_list.std() * 100, 2)}'
    f_score += f',{round(RuleFit_f1_score_list.mean() * 100, 2)}±{round(RuleFit_f1_score_list.std() * 100, 2)}'
    f_score += f',{round(rulecosi_f1_score_list.mean() * 100, 2)}±{round(rulecosi_f1_score_list.std() * 100, 2)}'
    f_score += f',{round(rules_f1_score_list.mean() * 100, 2)}±{round(rules_f1_score_list.std() * 100, 2)}'
    f_score += f',{round(rulefit_num_rules_list.mean() * 100, 2)}±{round(rulefit_num_rules_list.std() * 100, 2)}'
    f_score += f',{round(rules_num_rules_list.mean() * 100, 2)}±{round(rules_num_rules_list.std() * 100, 2)}'
    f_score += f',{round(rulecosi_num_rules_list.mean() * 100, 2)}±{round(rulecosi_num_rules_list.std() * 100, 2)}\n'

    print(f_score)
    f.write(f_score)

    accuracy_score = f'{filename},Accuracy-score,{round(cobertura_list.mean() * 100, 2)}±{round(cobertura_list.std() * 100, 2)}'
    accuracy_score += f',{round(tree_accuracy_list.mean() * 100, 2)}±{round(tree_accuracy_list.std() * 100, 2)}'
    accuracy_score += f',{round(ensemble_accuracy_list.mean() * 100, 2)}±{round(ensemble_accuracy_list.std() * 100, 2)}'
    accuracy_score += f',{round(RuleFit_accuracy_list.mean() * 100, 2)}±{round(RuleFit_accuracy_list.std() * 100, 2)}'
    accuracy_score += f',{round(rulecosi_accuracy_list.mean() * 100, 2)}±{round(rulecosi_accuracy_list.std() * 100, 2)}'
    accuracy_score += f',{round(rules_accuracy_list.mean() * 100, 2)}±{round(rules_accuracy_list.std() * 100, 2)}'
    accuracy_score += f',{round(rulefit_num_rules_list.mean() * 100, 2)}±{round(rulefit_num_rules_list.std() * 100, 2)}'
    accuracy_score += f',{round(rules_num_rules_list.mean() * 100, 2)}±{round(rules_num_rules_list.std() * 100, 2)}'
    accuracy_score += f',{round(rulecosi_num_rules_list.mean() * 100, 2)}±{round(rulecosi_num_rules_list.std() * 100, 2)}\n'

    print(accuracy_score)
    f.write(accuracy_score)

    precision_score = f'{filename},Precision-Score,{round(cobertura_list.mean() * 100, 2)}±{round(cobertura_list.std() * 100, 2)}'
    precision_score += f',{round(tree_precision_score_list.mean() * 100, 2)}±{round(tree_precision_score_list.std() * 100, 2)}'
    precision_score += f',{round(ensemble_precision_score_list.mean() * 100, 2)}±{round(ensemble_precision_score_list.std() * 100, 2)}'
    precision_score += f',{round(RuleFit_precision_score_list.mean() * 100, 2)}±{round(RuleFit_precision_score_list.std() * 100, 2)}'
    precision_score += f',{round(rulecosi_precision_score_list.mean() * 100, 2)}±{round(rulecosi_precision_score_list.std() * 100, 2)}'
    precision_score += f',{round(rules_precision_score_list.mean() * 100, 2)}±{round(rules_precision_score_list.std() * 100, 2)}'
    precision_score += f',{round(rulefit_num_rules_list.mean() * 100, 2)}±{round(rulefit_num_rules_list.std() * 100, 2)}'
    precision_score += f',{round(rules_num_rules_list.mean() * 100, 2)}±{round(rules_num_rules_list.std() * 100, 2)}'
    precision_score += f',{round(rulecosi_num_rules_list.mean() * 100, 2)}±{round(rulecosi_num_rules_list.std() * 100, 2)}\n'

    print(precision_score)
    f.write(precision_score)

    recall = f'{filename},Recall-Score,{round(cobertura_list.mean() * 100, 2)}±{round(cobertura_list.std() * 100, 2)}'
    recall += f',{round(tree_recall_list.mean() * 100, 2)}±{round(tree_recall_list.std() * 100, 2)}'
    recall += f',{round(ensemble_recall_list.mean() * 100, 2)}±{round(ensemble_recall_list.std() * 100, 2)}'
    recall += f',{round(RuleFit_recall_list.mean() * 100, 2)}±{round(RuleFit_recall_list.std() * 100, 2)}'
    recall += f',{round(rulecosi_recall_list.mean() * 100, 2)}±{round(rulecosi_recall_list.std() * 100, 2)}'
    recall += f',{round(rules_recall_list.mean() * 100, 2)}±{round(rules_recall_list.std() * 100, 2)}'
    recall += f',{round(rulefit_num_rules_list.mean() * 100, 2)}±{round(rulefit_num_rules_list.std() * 100, 2)}'
    recall += f',{round(rules_num_rules_list.mean() * 100, 2)}±{round(rules_num_rules_list.std() * 100, 2)}'
    recall += f',{round(rulecosi_num_rules_list.mean() * 100, 2)}±{round(rulecosi_num_rules_list.std() * 100, 2)}\n'

    print(recall)
    f.write(recall)

    return f_score, accuracy_score, precision_score, recall


def kfold_test(X, chi_square_percent_point_function, dataset, min_accuracy_coefficient,
               min_number_class_per_node, n_splits, n_repeats, scale_feature_coefficient, sorting_method,
               target_value_name, y):
    cobertura_list = []
    rules_accuracy_list = []
    rules_f1_score_list = []
    rules_precision_score_list = []
    rules_recall_list = []
    rules_roc_auc_score_list = []
    ensemble_accuracy_list = []
    ensemble_f1_score_list = []
    ensemble_precision_score_list = []
    ensemble_recall_list = []
    ensemble_roc_auc_score_list = []
    tree_accuracy_list = []
    tree_f1_score_list = []
    tree_precision_score_list = []
    tree_recall_list = []
    tree_roc_auc_score_list = []
    RuleFit_accuracy_list = []
    RuleFit_f1_score_list = []
    RuleFit_precision_score_list = []
    RuleFit_recall_list = []
    RuleFit_roc_auc_score_list = []
    rulecosi_accuracy_list = []
    rulecosi_f1_score_list = []
    rulecosi_precision_score_list = []
    rulecosi_recall_list = []
    rulecosi_roc_auc_score_list = []
    # number of rules
    rulefit_num_rules_list = []
    rules_num_rules_list = []
    rulecosi_num_rules_list = []

    repeated_kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    for train, test in repeated_kfold.split(X, y):
        custom_scorer = make_scorer(f1_score, greater_is_better=True)
        param_grid_tree = {
            'max_depth': [2, 3, 4, 5, 6],  # number of minimum samples required at a leaf node.
        }
        param_grid = {
            'n_estimators': [10, 25, 50, 100, 250, 500],  # being the number of trees in the forest.
            'max_depth': [2, 3, 4, 5, 6],  # number of minimum samples required at a leaf node.
        }

        # Random Forest
        clf_rf = GridSearchCV(
            # Evaluates the performance of different groups of parameters for a model based on cross-validation.
            RandomForestClassifier(),
            param_grid,  # dict of parameters.
            cv=5,  # Specified number of folds in the Cross-Validation(K-Fold).
            scoring=custom_scorer)

        # TREE
        clf_tree = GridSearchCV(
            # Evaluates the performance of different groups of parameters for a model based on cross-validation.
            DecisionTreeClassifier(),
            param_grid_tree,  # dict of parameters.
            cv=5,  # Specified number of folds in the Cross-Validation(K-Fold).
            scoring=custom_scorer)

        X_train = X.loc[train].to_numpy()
        y_train = y.loc[train].to_numpy()
        X_test = X.loc[test].to_numpy()
        y_test = y.loc[test].to_numpy()

        X_train_int = X_train.astype(int)
        y_train_int = y_train.astype(int)
        X_test_int = X_test.astype(int)
        y_test_int = y_test.astype(int)

        train_pandas_dataset = pd.DataFrame(data=np.c_[X_train, y_train],
                                            columns=list(dataset['feature_names']) + [target_value_name])

        rules = iPRules(
            feature_names=dataset.feature_names,
            target_value_name=dataset.target_names,
            chi_square_percent_point_function=chi_square_percent_point_function,
            scale_feature_coefficient=scale_feature_coefficient,
            min_accuracy_coefficient=min_accuracy_coefficient,
            min_number_class_per_node=min_number_class_per_node
        )
        # Fit model
        clf_rf.fit(X_train, y_train)
        ensemble = clf_rf.best_estimator_

        # RuleFit
        ruleFit = RuleFitClassifier()  # (tree_generator=ensemble)# alpha= [0.1, 1, 10, 100] include_linear= [True, False]

        # RULECOSI
        rulecosi = RuleCOSIClassifier(
            base_ensemble=ensemble,
            conf_threshold=0.9,
            cov_threshold=0.0,
            column_names=dataset.feature_names)

        # Fit model
        rules.fit(train_pandas_dataset, ensemble.feature_importances_)
        clf_tree.fit(X_train, y_train)
        tree = clf_tree.best_estimator_

        ruleFit.fit(X_train_int, y_train_int, feature_names=dataset.feature_names)
        rulecosi.fit(X_train, y_train)

        # Predict
        y_pred_test_ensemble = ensemble.predict(X_test)
        y_pred_test_rules = rules.predict(X_test, sorting_method=sorting_method)
        y_pred_test_tree = tree.predict(X_test)
        y_pred_test_RuleFit = ruleFit.predict(X_test_int)
        y_pred_test_rulecosi = rulecosi.predict(X_test)

        # DATASET CATEGORIZABLES
        np_array_rules = np.array(y_pred_test_rules)
        filter_indices = np.where(np_array_rules != None)[0]
        filtered_y_test = np.array(y_test)[filter_indices].astype('int64')
        filtered_y_test_int = np.array(y_test_int)[filter_indices].astype('int64')
        filtered_y_pred_test_ensemble = np.array(y_pred_test_ensemble)[filter_indices].astype('int64')
        filtered_y_pred_test_tree = np.array(y_pred_test_tree)[filter_indices].astype('int64')
        filtered_y_pred_test_RuleFit = np.array(y_pred_test_RuleFit)[filter_indices].astype('int64')
        filtered_y_pred_test_rules = np.array(y_pred_test_rules)[filter_indices].astype('int64')
        filtered_y_pred_test_rulecosi = np.array(y_pred_test_rulecosi)[filter_indices].astype('int64')

        if len(filter_indices) == 0:
            continue
        # SCORERS
        cobertura = len(filtered_y_pred_test_rules) / len(y_test)
        cobertura_list.append(cobertura)

        # Scores
        ensemble_accuracy, ensemble_f1_score, ensemble_precision_score, ensemble_recall, ensemble_roc_auc_score = generate_scores(
            filtered_y_test, filtered_y_pred_test_ensemble)
        ensemble_accuracy_list.append(ensemble_accuracy)
        ensemble_f1_score_list.append(ensemble_f1_score)
        ensemble_precision_score_list.append(ensemble_precision_score)
        ensemble_recall_list.append(ensemble_recall)
        ensemble_roc_auc_score_list.append(ensemble_roc_auc_score)

        tree_accuracy, tree_f1_score, tree_precision_score, tree_recall, tree_roc_auc_score = generate_scores(
            filtered_y_test, filtered_y_pred_test_tree)

        tree_accuracy_list.append(tree_accuracy)
        tree_f1_score_list.append(tree_f1_score)
        tree_precision_score_list.append(tree_precision_score)
        tree_recall_list.append(tree_recall)
        tree_roc_auc_score_list.append(tree_roc_auc_score)

        RuleFit_accuracy, RuleFit_f1_score, RuleFit_precision_score, RuleFit_recall, RuleFit_roc_auc_score = generate_scores(
            filtered_y_test_int, filtered_y_pred_test_RuleFit)
        RuleFit_accuracy_list.append(RuleFit_accuracy)
        RuleFit_f1_score_list.append(RuleFit_f1_score)
        RuleFit_precision_score_list.append(RuleFit_precision_score)
        RuleFit_recall_list.append(RuleFit_recall)
        RuleFit_roc_auc_score_list.append(RuleFit_roc_auc_score)

        rules_accuracy, rules_f1_score, rules_precision_score, rules_recall, rules_roc_auc_score = generate_scores(
            filtered_y_test, filtered_y_pred_test_rules)

        rules_accuracy_list.append(rules_accuracy)
        rules_f1_score_list.append(rules_f1_score)
        rules_precision_score_list.append(rules_precision_score)
        rules_recall_list.append(rules_recall)
        rules_roc_auc_score_list.append(rules_roc_auc_score)

        rulecosi_accuracy, rulecosi_f1_score, rulecosi_precision_score, rulecosi_recall, rulecosi_roc_auc_score = generate_scores(
            filtered_y_test, filtered_y_pred_test_rulecosi)

        rulecosi_accuracy_list.append(rulecosi_accuracy)
        rulecosi_f1_score_list.append(rulecosi_f1_score)
        rulecosi_precision_score_list.append(rulecosi_precision_score)
        rulecosi_recall_list.append(rulecosi_recall)
        rulecosi_roc_auc_score_list.append(rulecosi_roc_auc_score)

        rulefit_num_rules_list.append(len(ruleFit.rules_))
        rules_num_rules_list.append(len(rules.rules_))
        rulecosi_num_rules_list.append(len(rulecosi.simplified_ruleset_.rules))

    return np.array(cobertura_list), \
        np.array(RuleFit_accuracy_list), np.array(RuleFit_f1_score_list), np.array(
        RuleFit_precision_score_list), np.array(RuleFit_recall_list), np.array(RuleFit_roc_auc_score_list), \
        np.array(ensemble_accuracy_list), np.array(ensemble_f1_score_list), np.array(
        ensemble_precision_score_list), np.array(ensemble_recall_list), np.array(ensemble_roc_auc_score_list), \
        np.array(rules_accuracy_list), np.array(rules_f1_score_list), np.array(rules_precision_score_list), np.array(
        rules_recall_list), np.array(rules_roc_auc_score_list), \
        np.array(tree_accuracy_list), np.array(tree_f1_score_list), np.array(tree_precision_score_list), np.array(
        tree_recall_list), np.array(tree_roc_auc_score_list), \
        np.array(rulecosi_accuracy_list), np.array(rulecosi_f1_score_list), np.array(
        rulecosi_precision_score_list), np.array(rulecosi_recall_list), np.array(rulecosi_roc_auc_score_list), \
        np.array(rulefit_num_rules_list), np.array(rules_num_rules_list), np.array(rulecosi_num_rules_list)
