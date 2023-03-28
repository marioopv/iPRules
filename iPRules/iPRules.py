from copy import copy

import numpy as np
from scipy.stats import chi2_contingency, chi2
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

from iPRules.utils import FeatureComparer, Node, Pattern, chi_sq_node, concatenate_query, predict_unique_with_query


class iPRules(ClassifierMixin):

    def __init__(self,
                 base_ensemble,
                 feature_names,
                 target_value_name="target",
                 target_true=1,
                 target_false=0,
                 chi_square_probability=0.95,
                 scale_feature_coefficient=0.85,
                 min_accuracy_coefficient=0.9,
                 min_number_class_per_node=3,
                 epsilon_exp=0.0001
                 ):
        self.tree_ = None
        self.rules_ = []
        self.feature_importance_list = None
        self.most_important_features_ = None
        self.nodes_dict = {}  # TODO:
        self.nodes = []
        self.base_ensemble_ = base_ensemble
        self.feature_names = feature_names
        self.target_value_name = target_value_name
        self.target_true = target_true
        self.target_false = target_false
        self.target_class_positive = FeatureComparer(target_value_name, '==', self.target_false)
        self.target_class_negative = FeatureComparer(target_value_name, '==', self.target_true)
        self.chi_square_probability = chi_square_probability
        self.scale_feature_coefficient = scale_feature_coefficient
        self.min_accuracy_coefficient = min_accuracy_coefficient
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
    def chi2_values(self, matrix):
        matrix[matrix == 0] = 0.0001
        statistic, p_value, dof, expected_freq = chi2_contingency(matrix, correction=False)
        critical = chi2.ppf(self.chi_square_probability, dof)
        return statistic, p_value, dof, expected_freq, critical

    def get_top_important_features_list(self):
        """
        Obtiene las características más importantes en orden descendente
        :return:
        :param coefficient: Coeficiente entre 0 y 1 usado para obtener un % de las características más importantes.
        :param feature_names: Lista de los nombres de las columnas del dataset.
        :param feature_importances: Valor de importancia asociado a cada característica en el modelo entrenado.
        :return: Ordered feature list
        """
        # Indices de las características mas significativas ordenadas
        index = np.argsort(self.feature_importance_list)[::-1].tolist()
        max_coefficient = self.feature_importance_list[index[0]]  # Valor de la característica más importante
        coefficient_threshold = max_coefficient * (1 - self.scale_feature_coefficient)
        return [self.feature_names[x] for x in index if self.feature_importance_list[x] >= coefficient_threshold]

    def obtain_pattern_list_of_valid_nodes_with_pvalue(self):
        """
        Construct the list of rules based on the chi square of their sons
        @return: pattern_list_valid_nodes
        """
        pattern_list_valid_nodes = []
        visited_nodes = []  # Lista auxiliar para guardar los IDs de los nodos que ya han sido visitados.
        # Visita todos los nodos, y de aquellos que no sean el nodo principal y que tengan hijos, obtiene el chi-square de los hijos de ese nodo.
        for node in self.nodes:
            if node.PARENT_ID is None:
                continue
            parent_node = self.get_node(node.PARENT_ID)
            if parent_node.ID in visited_nodes:
                continue
            visited_nodes.append(parent_node.ID)
            # Obtiene la lista de IDs de sus nodos hermanos
            children = parent_node.children
            # En el caso de que ese nodo no sea un nodo hoja
            # (Recordar que solo busco nodos padre para calcular el chi-square del total de sus hijos)
            if len(children) > 0:
                aux_matrix = []
                for child_id in children:  # Access every single sibling node
                    aux_node = self.get_node(child_id)
                    aux_matrix.append([aux_node.number_positives, aux_node.number_negatives])
                # Se calcula el p valor de los hermanos en ese subnivel
                np_matrix = np.array(aux_matrix).astype(float).transpose()
                statistic, p_value, dof, expected_freq, critical = self.chi2_values(np_matrix)

                if True if abs(statistic) >= critical else False:  # TODO: DEFINE
                    # Set rules and last value to NONE
                    current_full_feature_comparer = copy(node.full_feature_comparer)
                    last_value = copy(current_full_feature_comparer[-1])
                    last_value.value = None
                    current_full_feature_comparer[-1] = last_value

                    # Si se encuentra una regla que puede tener un patrón, se incluye.
                    pattern = Pattern(target_value=None,
                                      full_feature_comparer=current_full_feature_comparer,
                                      p_value=p_value,
                                      number_target=None,
                                      feature_names=self.feature_names,
                                      number_all=None,
                                      target_accuracy=None)
                    pattern_list_valid_nodes.append(pattern)
        return pattern_list_valid_nodes

    def categorize_patterns(self, test_data, pattern_list_valid_nodes):
        """
        PSEUDO FIT
        :param test_data:
        :param pattern_list_valid_nodes:
        :return: list of rules
        """
        # Checks all combinations found and checks for both 0 and 1 in the last pathology to study both cases.
        index = 0
        while index < len(pattern_list_valid_nodes):
            rule = pattern_list_valid_nodes[index]
            for distinct_value in [self.target_true, self.target_false]:
                # UPDATE VALUES
                new_rule = copy(rule)
                last_value = copy(new_rule.full_feature_comparer[-1])
                last_value.value = distinct_value
                new_rule.full_feature_comparer[-1] = last_value

                number_negatives = self.count_query_negatives(test_data, new_rule.get_full_rule())
                number_positives = self.count_query_positives(test_data, new_rule.get_full_rule())
                number_all = number_positives + number_negatives
                # If this rule has existing cases in total in the training set, is included.
                if number_all > 0:
                    new_rule.number_all = number_all
                    # Checks if the combinations show a rule for negative/positives
                    proportion_positives = number_positives / number_all
                    # do not include rules with 0.5 prob
                    if proportion_positives == 0.5:
                        continue
                    if proportion_positives >= self.min_accuracy_coefficient:
                        new_rule.target_value = self.target_true
                        new_rule.number_target = number_positives
                        new_rule.target_accuracy = proportion_positives
                    else:

                        proportion_negatives = number_positives / number_all
                        new_rule.target_value = self.target_false
                        new_rule.number_target = number_negatives
                        new_rule.target_accuracy = proportion_negatives
                    self.rules_.append(new_rule)
            index += 1
        return self.rules_

    def predict_unique_with_query_positives(self, pandas_dataset, full_query):
        new_query = concatenate_query(full_query, self.target_class_positive.get_query())
        return predict_unique_with_query(pandas_dataset, new_query)

    def predict_unique_with_query_negatives(self, pandas_dataset, full_query):
        new_query = concatenate_query(full_query, self.target_class_negative.get_query())
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
        :param parent_node: node of the parent of current node
        :return:
        """
        if feature_index >= len(self.most_important_features_):
            # No hay más niveles
            return

        feature_comparer = FeatureComparer(self.most_important_features_[feature_index], '==', str(node_value))
        # Caso base para el que se considera como nodo padre de todos.
        if parent_node is None:
            # Create Node
            current_node = Node(ID=0,
                                PARENT_ID=None,
                                number_positives=self.count_query_negatives(dataset, ''),
                                number_negatives=self.count_query_positives(dataset, ''),
                                chi_sq_negative=0,
                                chi_sq_positive=0,
                                full_feature_comparer=[])
            # Incluye el nodo en la lista
            self.nodes.append(current_node)
            # Una vez creado el padre, se accede a la primera característica, que representaría el primer nivel.
            # Por cada posible valor que pueda tomar esa característica, se crea un hijo nodo de manera recursiva
            for node_value in dataset[self.most_important_features_[0]].unique():
                self.binary_tree_generator(dataset, node_value=node_value, parent_node=current_node)

        # Caso en el que el padre ya ha sido creado
        else:
            full_rule_query = concatenate_query(previous_full_query, feature_comparer.get_query())
            number_negatives = self.count_query_negatives(dataset, full_rule_query)
            number_positives = self.count_query_positives(dataset, full_rule_query)
            node_values_total = number_negatives + number_positives

            # Si el nodo se considera que no tiene los casos suficientes,
            # es descartado y el árbol no continúa en esa rama.
            if node_values_total >= self.min_number_class_per_node:

                # Los valores de muertes y supervivencias del padre se obtienen para calcular el chi-square
                # parent_number_negatives, parent_number_positives = self.get_ds_node(parent_id)
                parent_total = number_negatives + parent_node.number_positives
                chi_sq_negative, chi_sq_positive = chi_sq_node(number_negatives, parent_node.number_positives,
                                                               parent_total,
                                                               number_negatives, number_positives, node_values_total,
                                                               self.epsilon_exp)
                # Se le asigna la ID al nodo como la siguiente a la última utilizada.
                node_ID = self.nodes[-1].ID + 1
                current_node = Node(ID=node_ID,
                                    PARENT_ID=parent_node.ID,
                                    number_negatives=number_negatives,
                                    number_positives=number_positives,
                                    chi_sq_negative=chi_sq_negative,
                                    chi_sq_positive=chi_sq_positive,
                                    full_feature_comparer=parent_node.full_feature_comparer + [feature_comparer]
                                    )

                # Incluye el nodo en la lista
                self.nodes.append(current_node)

                # La ID del nodo es incluida en la lista de hijos del padre.
                self.nodes[parent_node.ID].children.append(node_ID)
                # self.add_child_to_parent(parent_node.ID, node_ID)

                # Por cada posible valor que pueda tomar esa característica, se crea un hijo nodo de manera recursiva
                for node_value in dataset[self.most_important_features_[feature_index]].unique():
                    new_feature_index = feature_index + 1
                    self.binary_tree_generator(dataset, previous_full_query=full_rule_query, node_value=node_value,
                                               feature_index=new_feature_index, parent_node=current_node)

    def fit(self, pandas_dataset):
        """
        Get list of top features and generate rules
        :param pandas_dataset:
        :return:
        """
        # Fit base model
        print("->Check Ensemble Model is fitted")
        check_is_fitted(self.base_ensemble_)

        print("->Extract feature importance list")
        # Feature Importance list
        self.feature_importance_list = self.base_ensemble_.feature_importances_
        # List of top % important features in the model are obtained. This % regulated by coefficient between [0,1].
        self.most_important_features_ = self.get_top_important_features_list()

        print("->Generate new tree based on list")
        # Genera el árbol binario y obtiene las combinaciones que indican que hay un patrón:
        self.binary_tree_generator(dataset=pandas_dataset)

        print("->Generate obtained patterns tree")
        pattern_list_valid_nodes = self.obtain_pattern_list_of_valid_nodes_with_pvalue()

        print("->Categorize patterns")
        self.categorize_patterns(pandas_dataset, pattern_list_valid_nodes)

        return self

    def predict(self, X):
        """
        Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """

        predictions = []
        for x in X:
            categorize_rule = None
            for rule in sorted(self.rules_, key=lambda r: r.target_accuracy):
                prediction = rule.Predict(x)
                if prediction is not None:
                    predictions.append(prediction)
                    categorize_rule = True
                    break
            if categorize_rule is None:
                predictions.append(None)

        return predictions

    def __str__(self):
        display = '> ------------------------------\n'
        display += '> iPRules (not ordered):\n'
        display += f'> Number of Rules {len(self.rules_)}:\n'
        display += '> ------------------------------\n'
        for num in range(len(self.rules_)):
            display += f'Rule {num}:\n {self.rules_[num]}'

        return display
