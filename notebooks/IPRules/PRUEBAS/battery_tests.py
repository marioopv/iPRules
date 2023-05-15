from notebooks.IPRules.read_datasets import dataset_names, read_dataset
from notebooks.IPRules.test_utils import generate_battery_test

# Different values
n_splits = 10
n_repeats = 3
# chi_square_percent_point_function_list = [0.95, 0.97, 0.99]
chi_square_percent_point_function_list = [0.99]
scale_feature_coefficient = 0.01
min_accuracy_coefficient = 0.95
# min_number_class_per_node_list = [1, 3, 5, 7, 10]
min_number_class_per_node_list = [3, 5, 7, 1, 10]
sorting_method = "target_accuracy"

path = f'../..'

results_file_name = f'{path}/Results/battery_test_all_nodes_credit.csv'

f = open(results_file_name, "w")
file_header = f'Dataset, scorer, Coverage, DT, RF, RF+RFIT, RF+RFIT num rules, RF+RC, RF+RC num rules, RF+Rules, RF+Rules num rules\n'
print(file_header)
f.write(file_header)
for name in dataset_names:
    dataset_path_name = f'{path}/data/{name}.csv'
    X, y, dataset, target_value_name, pandas_dataset = read_dataset(name, dataset_path_name)

    for chi_square_percent_point_function in chi_square_percent_point_function_list:
        for min_number_class_per_node in min_number_class_per_node_list:
            f_score, accuracy_score, precision_score, recall = generate_battery_test(f, name, X, y, dataset,
                                                                                 target_value_name, n_splits, n_repeats,
                                                                                 chi_square_percent_point_function,
                                                                                 scale_feature_coefficient,
                                                                                 min_accuracy_coefficient,
                                                                                 min_number_class_per_node,
                                                                                 sorting_method)

f.close()
