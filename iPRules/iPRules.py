import re

import numpy as np
from scipy.stats import chi2_contingency, chi2
from sklearn.base import ClassifierMixin
from sklearn.tree import BaseDecisionTree
from sklearn.tree._tree import Tree
from sklearn.utils.validation import check_is_fitted


class Node:
    def __init__(self, ID, PARENT_ID, number_negatives, number_positives, chi_sq_negative, chi_sq_positive, full_query,
                 feature_comparer, full_feature_comparer=[]):
        self.ID = ID  # int
        self.PARENT_ID = PARENT_ID  # int
        self.number_negatives = number_negatives  # int
        self.number_positives = number_positives  # int
        self.chi_sq_negative = chi_sq_negative  # float
        self.chi_sq_positive = chi_sq_positive  # float
        self.full_query = full_query  # str
        self.feature_comparer = feature_comparer  # str
        self.full_feature_comparer = full_feature_comparer
        self.children = []  # list of int. Contiene todos los IDs de los hijos


class FeatureComparer:
    def __init__(self, feature_name, comparer, value):
        self.feature_name = feature_name
        self.comparer = comparer
        self.value = value
        self.query = f'{self.feature_name} {self.comparer} {self.value}'


def predict_unique_with_query(pandas_dataset, full_query):
    return pandas_dataset.query(full_query)


def concatenate_query(previous_full_query, rule_query):
    # A la query auxiliar se le incluye la característica del nodo junto con el valor asignado
    full_rule_query = rule_query

    if previous_full_query != '':
        full_rule_query = previous_full_query + ' & ' + full_rule_query

    return full_rule_query


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


class iPRules(ClassifierMixin, BaseDecisionTree):

    def __init__(self,
                 base_ensemble,
                 feature_names,
                 target_value_name="target",
                 target_true="1",
                 target_false="0",
                 chi_square_probability=0.95,
                 scale_feature_coefficient=0.85,
                 min_number_class_per_node=3,
                 epsilon_exp=0.0001
                 ):
        self.tree_ = None
        self.positive_patterns = []
        self.negative_patterns = []
        self.rules_ = []
        self.obtain_patterns = None
        self.feature_importance_list = None
        self.most_important_features = None
        self.nodes_dict = {}
        self.nodes = []
        self.base_ensemble = base_ensemble
        self.feature_names = feature_names
        self.target_value_name = target_value_name
        self.target_true = target_true
        self.target_false = target_false
        self.target_class_positive = FeatureComparer(target_value_name, '==', self.target_false)
        self.target_class_negative = FeatureComparer(target_value_name, '==', self.target_true)
        self.chi_square_probability = chi_square_probability
        self.scale_feature_coefficient = scale_feature_coefficient
        self.min_number_class_per_node = min_number_class_per_node
        self.epsilon_exp = epsilon_exp

    def add_child_to_parent(self, parent_id, child_id):
        i = 0
        found = False
        while i < len(self.nodes) and not found:  # Busca el nodo padre
            if parent_id is self.nodes[i].ID:
                self.nodes[i].children.append(child_id)
                found = True
            i += 1

    def get_node(self, ID):
        i = -1
        while i < len(self.nodes):
            if ID is self.nodes[i].ID:
                return self.nodes[i]
            else:
                i += 1

    def is_leaf(self, ID):
        for node in self.nodes:
            if node.PARENT_ID == ID:
                return True
        return False

    def get_ds_node(self, ID):
        """
        Devuelve número de fallecimientos y supervivencias de ese nodo.
        :param ID:
        :return:
        """
        found = False
        index = 0
        i = 0
        while i < len(self.nodes) and not found:
            if self.nodes[i].ID is ID:
                index = i
                found = True
            i += 1
        return self.nodes[index].number_negatives, self.nodes[index].number_positives

    # Calcula los coeficientes de chi-square usando los valores de muertes y
    # supervivencias del nodo en cuestión y del nodo padre

    def has_pattern(self, matrix):
        matrix[matrix == 0] = 0.0001
        # print('Matrix is:',matrix)
        stat, p, dof, expected = chi2_contingency(matrix, correction=False)
        critical = chi2.ppf(self.chi_square_probability, dof)
        return True if abs(stat) >= critical else False

    def get_top_important_features_list(self):
        """
        Obtiene las características más importantes en orden descendente
        :return:
        :param coefficient: Coeficiente entre 0 y 1 usado para obtener un % de las características más importantes.
        :param feature_names: Lista de los nombres de las columnas del dataset.
        :param feature_importances: Valor de importancia asociado a cada característica en el modelo entrenado.
        :return: Ordered feature list
        """
        index = np.argsort(self.feature_importance_list)[
                ::-1].tolist()  # Indices de las características mas significativas
        max_coefficient = self.feature_importance_list[index[0]]  # Valor de la característica más importante
        coefficient_threshold = max_coefficient * (1 - self.scale_feature_coefficient)
        return [self.feature_names[x] for x in index if self.feature_importance_list[x] >= coefficient_threshold]

    def obtain_patterns_tree(self):
        """

        (
            rule : ,
            value:
        )
        :return:
        """
        patterns = set()
        visited_nodes = []  # Lista auxiliar para guardar los IDs de los nodos que ya han sido visitados.
        # Visita todos los nodos, y de aquellos que no sean el nodo principal y que tengan hijos, obtiene el chi-square de los hijos de ese nodo.
        for node in self.nodes:
            if node.PARENT_ID is not None:
                parent_node = self.get_node(node.PARENT_ID)
                if parent_node.ID not in visited_nodes:  # Evita que el nodo padre sea contado múltiples veces, ya que se calcula el chi-square de los hijos 1 sola vez
                    visited_nodes.append(parent_node.ID)
                    children = parent_node.children  # Obtiene la lista de IDs de sus nodos hermanos
                    if len(children) > 0:  # En el caso de que ese nodo no sea un nodo hoja (Recordar que solo busco nodos padre para calcular el chi-square del total de sus hijos)
                        aux_matrix = []
                        aux_query = ''
                        for child_id in children:  # Access every single sibling node
                            aux_node = self.get_node(child_id)
                            aux_matrix.append([aux_node.number_positives, aux_node.number_negatives])
                            # Se parsea la query para representar el nivel.
                            aux_query = re.search("(.*) ==.*", aux_node.full_query)
                        # if has_enough_cases:  # Se calcula el p valor de los hermanos en ese subnivel
                        np_matrix = np.array(aux_matrix).astype(float).transpose()
                        if self.has_pattern(np_matrix):  # TODO:?? and :
                            patterns.add(
                                aux_query[1])  # Si se encuentra una regla que puede tener un patrón, se incluye.

        self.obtain_patterns = list(patterns)
        return patterns

    def categorize_patterns(self, test_data, coefficient):
        """
        PSEUDO FIT
        :param test_data:
        :param patterns:
        :param coefficient:
        :return:
                (
            rule : ,
            value:
        )
        """
        # TODO: DEFINE RULES
        # OVERLAP???
        # Checks all combinations found and checks for both 0 and 1 in the last pathology to study both cases.
        for pattern in self.obtain_patterns:
            values = [0, 1]
            for j in values: #TODO: Get all different values of pathology
                full_rule_query = pattern + ' == ' + str(j)  # Adds the 0 or 1 to the last pathology #TODO: WHY
                number_negatives = self.count_query_negatives(test_data, full_rule_query)
                number_positives = self.count_query_positives(test_data, full_rule_query)
                all_cases = number_positives + number_negatives
                # If this pattern has existing cases in total in the training set, is included.
                if all_cases > 0:
                    # Checks if the combinations show a pattern for negative/positives
                    if (number_negatives / all_cases) >= coefficient:
                        self.negative_patterns.append([pattern + ' == ' + str(j), number_negatives, all_cases])
                    elif (number_positives / all_cases) >= coefficient:
                        self.positive_patterns.append([pattern + ' == ' + str(j), number_positives, all_cases])

        return self.negative_patterns, self.positive_patterns

    def predict_unique_with_query_positives(self, pandas_dataset, full_query):
        new_query = concatenate_query(full_query, self.target_class_positive.query)
        return predict_unique_with_query(pandas_dataset, new_query)

    def predict_unique_with_query_negatives(self, pandas_dataset, full_query):
        new_query = concatenate_query(full_query, self.target_class_negative.query)
        return predict_unique_with_query(pandas_dataset, new_query)

    def count_query_positives(self, pandas_dataset, full_query):
        return len(self.predict_unique_with_query_positives(pandas_dataset, full_query))

    def count_query_negatives(self, pandas_dataset, full_query):
        return len(self.predict_unique_with_query_negatives(pandas_dataset, full_query))

    def binary_tree_generator(self,
                              dataset,
                              previous_full_query='',
                              node_value=0,
                              feature_index=0,
                              parent_node=None):
        """
        Función recursiva encargada de generar el árbol de nodos con sus respectivas queries y obtener en cada nodo la query y el número de fallecimientos y supervivencias de cada uno.

        :param dataset: Pandas DataFrame. Dataset con las filas para obtener el número de fallecimientos y defunciones usando cada query.
        :param previous_full_query: Variable auxiliar en la que se irá anexando las características junto con sus valores para generar las queries.
        :param node_value: Representa el valor de la característica en ese nodo en concreto.
        :param feature_index: índice auxiliar de la lista de características
        :param parent_id:
        :param parent_node: node of the parent of current node
        :return:


        (
            rule :
            comparer:
            target_value:
        )

        GENERATE QUERY FROM list of

                negative = len(test_data.query(query + self.target_class_negative.query))
                positive = len(test_data.query(query + self.target_class_positive.query))
        """
        # En el caso de que queden características para ampliar el nivel:
        if feature_index < len(self.most_important_features):

            feature_comparer = FeatureComparer(self.most_important_features[feature_index], '==', str(node_value))
            # Caso base para el que se considera como nodo padre de todos.
            if parent_node is None:
                # Create Node
                current_node = Node(ID=0,
                                    PARENT_ID=None,
                                    number_positives=self.count_query_negatives(dataset, ''),
                                    number_negatives=self.count_query_positives(dataset, ''),
                                    chi_sq_negative=0,
                                    chi_sq_positive=0,
                                    full_query=None,  # El padre tampoco tiene query, osea que se deja a 0.
                                    feature_comparer=feature_comparer,
                                    full_feature_comparer=[feature_comparer])
                # Incluye el nodo en la lista
                self.nodes.append(current_node)
                # Una vez creado el padre, se accede a la primera característica, que representaría el primer nivel.
                # Por cada posible valor que pueda tomar esa característica, se crea un hijo nodo de manera recursiva
                for node_value in dataset[self.most_important_features[0]].unique():
                    self.binary_tree_generator(dataset, node_value=node_value, parent_node=current_node)

            # Caso en el que el padre ya ha sido creado
            else:
                full_rule_query = concatenate_query(previous_full_query, feature_comparer.query)
                number_negatives = self.count_query_negatives(dataset, full_rule_query)
                number_positives = self.count_query_positives(dataset, full_rule_query)
                node_total = number_negatives + number_positives

                # Si el nodo se considera que no tiene los casos suficientes, es descartado y el árbol no continúa en esa rama.
                if node_total >= self.min_number_class_per_node:

                    # Los valores de muertes y supervivencias del padre se obtienen para calcular el chi-square
                    # parent_number_negatives, parent_number_positives = self.get_ds_node(parent_id)
                    parent_total = number_negatives + parent_node.number_positives
                    chi_sq_negative, chi_sq_positive = chi_sq_node(number_negatives, parent_node.number_positives,
                                                                   parent_total,
                                                                   number_negatives, number_positives, node_total,
                                                                   self.epsilon_exp)
                    # Se le asigna la ID al nodo como la siguiente a la última utilizada.
                    node_ID = self.nodes[-1].ID + 1
                    current_node = Node(ID=node_ID,
                                        PARENT_ID=parent_node.ID,
                                        number_negatives=number_negatives,
                                        number_positives=number_positives,
                                        chi_sq_negative=chi_sq_negative,
                                        chi_sq_positive=chi_sq_positive,
                                        full_query=full_rule_query,
                                        feature_comparer=feature_comparer,
                                        full_feature_comparer=parent_node.full_feature_comparer + [feature_comparer]
                                        )
                    # Incluye el nodo en la lista
                    self.nodes.append(current_node)

                    # La ID del nodo es incluida en la lista de hijos del padre.
                    self.nodes[parent_node.ID].children.append(node_ID)
                    # self.add_child_to_parent(parent_node.ID, node_ID)

                    # Por cada posible valor que pueda tomar esa característica, se crea un hijo nodo de manera recursiva
                    for node_value in dataset[self.most_important_features[feature_index]].unique():
                        new_feature_index = feature_index + 1
                        self.binary_tree_generator(dataset, previous_full_query=full_rule_query, node_value=node_value,
                                                   feature_index=new_feature_index, parent_node=current_node)

    def fit(self, pandas_dataset, X_train, y_train):
        """
        Get list of top features and generate rules
        :param pandas_dataset:
        :return:
        """
        # Fit base model
        print("Fit Ensemble Model")
        self.base_ensemble.fit(X_train, y_train)

        print("Extract feature importance list")
        # Feature Importance list
        self.feature_importance_list = self.base_ensemble.feature_importances_
        # List of top % important features in the model are obtained. This % regulated by coefficient between [0,1].
        self.most_important_features = self.get_top_important_features_list()

        print("Generate new tree based on list")
        # Genera el árbol binario y obtiene las combinaciones que indican que hay un patrón:
        self.binary_tree_generator(dataset=pandas_dataset)

        print("Generate obtained patterns tree")
        self.obtain_patterns_tree()
        # self.tree_generator(dataset=pandas_dataset)
        return self

    def tree_generator(self):
        # TODO:
        self.tree_ = Tree(
            self.n_features_in_,
            # TODO: tree shouldn't need this in this case
            np.array([1] * self.n_outputs_, dtype=np.intp),
            self.n_outputs_,
        )

    def predict_proba(self, X, check_input=True):

        # TODO: ALL
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)

        if self.n_outputs_ == 1:
            proba = proba[:, : self.n_classes_]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer

            return proba

        else:
            all_proba = []

            for k in range(self.n_outputs_):
                proba_k = proba[:, k, : self.n_classes_[k]]
                normalizer = proba_k.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba_k /= normalizer
                all_proba.append(proba_k)

            return all_proba

    def predict_log_proba(self, X):
        # TODO: ALL
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba

    def __str__(self):
        display = '> ------------------------------\n'
        display += '> iPRules:\n'
        display += f'> Number of patterns {len(self.obtain_patterns)}:\n'
        display += '> ------------------------------\n'
        for num in range(len(self.obtain_patterns)):
            display += f'Regla {num}: {self.obtain_patterns[num]}\n'
        return display
