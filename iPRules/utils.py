import numpy as np


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

class Node:
    def __init__(self,
                 ID,
                 PARENT_ID,
                 number_negatives,
                 number_positives,
                 full_feature_comparer=[]):
        self.ID = ID  # int
        self.PARENT_ID = PARENT_ID  # int
        self.number_negatives = number_negatives  # int
        self.number_positives = number_positives  # int
        self.full_feature_comparer = full_feature_comparer
        self.children = []  # list of int. Contiene todos los IDs de los hijos

    def get_full_query(self):
        full_query = ''
        for feature_comparer in self.full_feature_comparer:
            full_query = concatenate_query(full_query, feature_comparer.get_query())
        return full_query

    def __str__(self):
        display = '> ------------------------------\n'
        display += f'>** Node ID: {self.ID} '
        display += f'** Numer of comparisons: {len(self.full_feature_comparer)}:\n'
        display += '> ------------------------------\n'
        for feature_comparer in self.full_feature_comparer:
            display += f'\t{feature_comparer}\n'
        return display


class FeatureComparer:
    def __init__(self, feature_name, comparer, value):
        self.feature_name = feature_name
        self.comparer = comparer
        self.value = value

    def __str__(self):
        return self.get_query()

    def get_query(self):
        return f'{self.feature_name} {self.comparer} {str(self.value)}'

    def unitary_loc(self, dataset):
        return dataset.loc[dataset[self.feature_name] == self.value]


class Pattern:
    def __init__(self,
                 target_value,
                 feature_names,
                 full_feature_comparer,
                 chi2_statistic,
                 p_value,
                 chi2_critical_value,
                 expected_freq,
                 number_target,
                 number_all,
                 target_accuracy):
        self.target_value = target_value  # str
        self.p_value = p_value  # str
        self.chi2_statistic = chi2_statistic  # str
        self.chi2_critical_value = chi2_critical_value  # str
        self.expected_freq = expected_freq  # str
        self.full_feature_comparer = full_feature_comparer  # Node
        self.number_target = number_target
        self.number_all = number_all
        self.target_accuracy = target_accuracy
        self.feature_names = feature_names

    def get_complexity(self):
        return len(self.full_feature_comparer)

    def get_full_rule(self):
        full_query = ''
        for comparer in self.full_feature_comparer:
            full_query = concatenate_query(full_query, f'{comparer.get_query()}')
        return full_query

    def Predict(self, data_array):
        for comparer in self.full_feature_comparer:
            index = np.where(self.feature_names == comparer.feature_name)[0][0]
            if float(comparer.value) != float(data_array[index]):
                return None
        return self.target_value

    def __str__(self):
        display = '> ------------------------------\n'
        display += f' ** Target value: {self.target_value}'
        display += f' ** Target: {self.number_target}'
        display += f' ** Total: {self.number_all}'
        display += f' ** Accuracy: {self.target_accuracy}'
        display += f' ** Complexity: {self.get_complexity()}'
        display += f' ** Chi2 critical_value: {self.chi2_critical_value}'
        display += f' ** P_value: {self.p_value}\n'
        display += f'\t Query: {self.get_full_rule()}\n'
        display += '> ------------------------------\n'
        return display


def concatenate_query_comparer(full_feature_comparer):
    query = ''
    for g in full_feature_comparer:
        query = concatenate_query(query, g)
    return query


def chunk_query(dataset_filtered, new_query):
    if not "&" in new_query:
        dataset_filtered = predict_unique_with_query(dataset_filtered, new_query)
    for group in divide_chunks(new_query.split("&"), 31):
        group_query = concatenate_query_comparer(group)
        dataset_filtered = predict_unique_with_query(dataset_filtered, group_query)
    return dataset_filtered

def predict_unique_with_query(dataset, full_query):
    return dataset.query(full_query, engine='python')  #slower otherwise limit of 32 variables
    #return dataset.query(full_query)


def concatenate_query(previous_full_query, rule_query):
    # A la query auxiliar se le incluye la característica del nodo junto con el valor asignado
    if previous_full_query != '':
        return f'{previous_full_query}  &  {rule_query}'
    return f'{rule_query}'
