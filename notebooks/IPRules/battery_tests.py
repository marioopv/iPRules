from notebooks.IPRules.read_datasets import dataset_names, read_dataset
from notebooks.IPRules.test_utils import generate_battery_test

# Different values
n_splits = 10
n_repeats = 3
chi_square_percent_point_function = 0.95
scale_feature_coefficient = 0.01
min_accuracy_coefficient = 0.9
min_number_class_per_node = 3
sorting_method = "target_accuracy"

path = f'../..'

results_file_name = f'{path}/Results/battery_test_all.csv'


f = open(results_file_name, "w")
file_header = f'Dataset, scorer, Coverage, DT, RF, RF+RFIT, RF+RC, RF+Rules, RF+RFIT num rules, RF+Rules num rules, RF+RC num rules\n'
print(file_header)
f.write(file_header)
for name in dataset_names:
    dataset_path_name = f'{path}/data/{name}.csv'
    X, y, dataset, target_value_name, pandas_dataset = read_dataset(name, dataset_path_name)


    f_score, accuracy_score, precision_score, recall = generate_battery_test(f, name, X, y, dataset,
                                                                             target_value_name, n_splits, n_repeats,
                                                                             chi_square_percent_point_function,
                                                                             scale_feature_coefficient,
                                                                             min_accuracy_coefficient,
                                                                             min_number_class_per_node,
                                                                             sorting_method)

f.close()
