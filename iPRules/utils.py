class Node:
    def __init__(self,
                 ID,
                 PARENT_ID,
                 number_negatives,
                 number_positives,
                 chi_sq_negative,
                 chi_sq_positive,
                 full_feature_comparer=[]):
        self.ID = ID  # int
        self.PARENT_ID = PARENT_ID  # int
        self.number_negatives = number_negatives  # int
        self.number_positives = number_positives  # int
        self.chi_sq_negative = chi_sq_negative  # float
        self.chi_sq_positive = chi_sq_positive  # float
        self.full_feature_comparer = full_feature_comparer
        self.children = []  # list of int. Contiene todos los IDs de los hijos

    def get_full_query(self):
        full_query = ''
        for feature_comparer in self.full_feature_comparer:
            full_query = concatenate_query(full_query, feature_comparer.get_query())
        return full_query

    def __str__(self):
        display = '> ------------------------------\n'
        display += f'> Node ID: {self.ID}:\n'
        display += f'> Numer of comparisons: {len(self.full_feature_comparer)}:\n'
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
        return f'{self.feature_name} {self.comparer} {self.value}'


class Pattern:
    def __init__(self,
                 target_value,
                 feature_names,
                 full_feature_comparer,
                 p_value,
                 number_target,
                 number_all,
                 target_accuracy):
        self.target_value = target_value  # str
        self.p_value = p_value  # str
        self.full_feature_comparer = full_feature_comparer  # Node
        self.number_target = number_target
        self.number_all = number_all
        self.target_accuracy = target_accuracy
        self.feature_names = feature_names
        self.complexity = len(full_feature_comparer)

    def get_full_rule(self):
        full_query = ''
        for comparer in self.full_feature_comparer:
            full_query = concatenate_query(full_query, f'{comparer.get_query()}')
        return full_query

    def Predict(self, data_array):
        for comparer in self.full_feature_comparer:
            index = self.feature_names.where(comparer.feature_name)[0][0]
            if comparer.value != data_array[index]:
                return None

        return self.target_value

    def __str__(self):
        display = '> ------------------------------\n'
        display += f'> Numer of patterns: {len(self.full_feature_comparer)}:\n'
        display += f'> Target value: {self.target_value}\n' \
                   f'> P_value: {self.p_value}\n' \
                   f'> number_target: {self.number_target}\n' \
                   f'> number_all: {self.number_all}\n' \
                   f'> target_accuracy: {self.target_accuracy}\n' \
                   f'> complexity: {self.complexity}\n'
        display += '> ------------------------------\n'
        for comparer in self.full_feature_comparer:
            display += f'\t\t --- {comparer.get_query()}\n'
        display += '> ------------------------------\n'
        display += f'\tQuery: {self.get_full_rule()}\n'
        display += '> ------------------------------\n'
        return display


def predict_unique_with_query(pandas_dataset, full_query):
    return pandas_dataset.query(full_query)


def concatenate_query(previous_full_query, rule_query):
    """

    @param previous_full_query:
    @param rule_query:
    @return: query
    """
    # A la query auxiliar se le incluye la característica del nodo junto con el valor asignado
    if previous_full_query != '':
        return f'{previous_full_query}  &  {rule_query}'
    return f'{rule_query}'


def chi_sq_node(parent_number_negatives, parent_number_positives, parent_total,
                number_negatives, number_positives, node_total,
                epsilon_exp):
    """
    Calcula los coeficientes de chi-square usando los valores de muertes y
    supervivencias del nodo en cuestión y del nodo padre
    :return:
    :param parent_number_negatives:
    :param parent_number_positives:
    :param number_negatives:
    :param number_positives:
    :param epsilon_exp:Coeficiente usado para evitar que la bondad de ajuste sea 0
    :return: chisq_negatives, chisq_positives
    @type parent_number_negatives: object
    @param parent_number_negatives:
    @param node_total:
    """

    # Expected values
    expected_negatives = parent_number_negatives * node_total / parent_total
    expected_positives = parent_number_positives * node_total / parent_total

    if expected_negatives == 0:
        expected_negatives = epsilon_exp
    if expected_positives == 0:
        expected_positives = epsilon_exp
    # Los valores de chi-square
    chis_square_negatives = ((number_negatives - expected_negatives) ** 2) / expected_negatives
    chis_square_positives = ((number_positives - expected_positives) ** 2) / expected_positives

    return chis_square_negatives, chis_square_positives
