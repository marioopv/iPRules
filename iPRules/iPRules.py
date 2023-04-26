import time
import copy

import numpy as np
from scipy.stats import chi2_contingency, chi2
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn import preprocessing
from iPRules.utils import FeatureComparer, Node, Pattern, concatenate_query, predict_unique_with_query, divide_chunks, \
    concatenate_query_comparer, chunk_query


def plot_features(X_train_minmax):
    import matplotlib.pyplot as plt
    # data to be plotted
    # plotting

    plt.title("Features MinMaxScaler")
    plt.xlabel("X MinMaxScaler")
    plt.ylabel("Features")
    plt.plot(X_train_minmax, color="green")
    #plt.yticks([0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.show()


class iPRules(ClassifierMixin):

    def __init__(self,
                 feature_names,
                 target_value_name="target",
                 display_logs=False,
                 display_features=False,
                 target_true=1,
                 target_false=0,
                 chi_square_percent_point_function=0.95,
                 scale_feature_coefficient=0.2,
                 min_accuracy_coefficient=0.9,
                 min_number_class_per_node=3
                 ):
        self.rules_ = []
        self.feature_importance_list = None
        self.most_important_features_ = None
        self.nodes_dict = {}
        self.nodes_dict_ids = []
        self.pattern_list_valid_nodes = []
        self.feature_names = feature_names
        self.target_value_name = target_value_name
        self.target_true = target_true
        self.target_false = target_false
        self.target_class_positive = FeatureComparer(target_value_name, '==', self.target_true)
        self.target_class_negative = FeatureComparer(target_value_name, '==', self.target_false)
        self.chi_square_percent_point_function = chi_square_percent_point_function
        self.scale_feature_coefficient = scale_feature_coefficient
        self.min_accuracy_coefficient = min_accuracy_coefficient
        self.min_number_class_per_node = min_number_class_per_node
        self.display_features = display_features
        self.display_logs = display_logs

    def get_node(self, ID):
        return self.nodes_dict[ID]

    def parent_relation_matrix(self, children):
        # (Recordar que solo busco nodos padre para calcular el chi-square del total de sus hijos)
        aux_matrix = []
        for child_id in children:  # Access every single sibling node
            aux_node = self.get_node(child_id)
            aux_matrix.append([aux_node.number_positives, aux_node.number_negatives])
        # Se calcula el p valor de los hermanos en ese subnivel
        return np.array(aux_matrix).astype(float).transpose()

    # Calcula los coeficientes de chi-square usando los valores de muertes y
    # supervivencias del nodo en cuestión y del nodo padre
    def chi2_values(self, children):
        # https://matteocourthoud.github.io/post/chisquared/
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
        # https://towardsdatascience.com/gentle-introduction-to-chi-square-test-for-independence-7182a7414a95#92ef

        matrix = self.parent_relation_matrix(children)
        matrix[matrix == 0] = 0.0001
        chi2_results = chi2_contingency(matrix, correction=False)
        chi2_critical_value = chi2.ppf(self.chi_square_percent_point_function, chi2_results.dof)

        return chi2_results.statistic, chi2_results.pvalue, chi2_results.dof, chi2_results.expected_freq, chi2_critical_value

    def get_top_important_features_list(self, feature_importances):
        """
        Obtiene las características más importantes en orden descendente
        :return:
        :param coefficient: Coeficiente entre 0 y 1 usado para obtener un % de las características más importantes.
        :param feature_names: Lista de los nombres de las columnas del dataset.
        :param feature_importances: Valor de importancia asociado a cada característica en el modelo entrenado.
        :return: Ordered feature list
        """

        if self.display_logs:
            print("->Extract feature importance list")

        # Feature Importance list
        self.feature_importance_list = feature_importances

        # Indices de las características mas significativas ordenadas
        index = np.argsort(self.feature_importance_list)[::-1].tolist()
        max_coefficient = self.feature_importance_list[index[0]]  # Valor de la característica más importante

        X_train_minmax = self.nomalized_features()
        if self.display_features:
            plot_features(X_train_minmax)

        # coefficient_threshold = max_coefficient * (1 - self.scale_feature_coefficient)
        # self.most_important_features_ = [self.feature_names[x] for x in index if self.feature_importance_list[x] >= coefficient_threshold]

        self.most_important_features_ = [self.feature_names[x] for x in index if
                                         X_train_minmax[x] >= self.scale_feature_coefficient]

        if self.display_logs:
            print(f'\t Original features {len(self.feature_importance_list)}')
            print(f'\t Selected features {len(self.most_important_features_)}')
            print(
                f'\t Percentage of selected rules: {100 * len(self.most_important_features_) / len(self.feature_importance_list)} %')

    def nomalized_features(self):
        return preprocessing.MinMaxScaler().fit_transform(self.feature_importance_list.reshape(-1, 1))

    def obtain_pattern_list_of_valid_nodes_with_pvalue(self):
        """
        Construct the list of rules based on the chi square of their sons
        @return: pattern_list_valid_nodes
        """

        start_time = time.time()
        if self.display_logs:
            print("->Generate obtained patterns tree")

        visited_nodes = []  # Lista auxiliar para guardar los IDs de los nodos que ya han sido visitados.
        # Visita todos los nodos, y de aquellos que no sean el nodo principal y que tengan hijos, obtiene el chi-square de los hijos de ese nodo.
        for key, node in self.nodes_dict.items():
            if node.PARENT_ID is None:
                continue
            parent_node = self.get_node(node.PARENT_ID)
            if parent_node.ID in visited_nodes:
                continue
            visited_nodes.append(parent_node.ID)
            # Obtiene la lista de IDs de sus nodos hermanos
            children = parent_node.children
            # En el caso de que ese nodo no sea un nodo hoja
            if len(children) > 0:
                chi2_statistic, p_value, degrees_of_freedom, expected_freq, chi2_critical_value = self.chi2_values(children)

                if chi2_statistic > chi2_critical_value:
                    # Set rules and last value to NONE
                    current_full_feature_comparer = copy.deepcopy(node.full_feature_comparer)
                    current_full_feature_comparer[-1].value = None

                    # Si se encuentra una regla que puede tener un patrón, se incluye.
                    pattern = Pattern(target_value=None,
                                      full_feature_comparer=current_full_feature_comparer,
                                      p_value=p_value,
                                      chi2_statistic=chi2_statistic,
                                      chi2_critical_value=chi2_critical_value,
                                      expected_freq=expected_freq,
                                      number_target=None,  # define later
                                      feature_names=self.feature_names,
                                      number_all=None,
                                      target_accuracy=None)
                    self.pattern_list_valid_nodes.append(pattern)

        elapsed_time = time.time() - start_time
        if self.display_logs:
            print(
                f"Elapsed time to compute the obtain_pattern_list_of_valid_nodes_with_pvalue: {elapsed_time:.3f} seconds")

    def categorize_patterns(self, test_data):
        """
        PSEUDO FIT
        :param test_data:
        :param pattern_list_valid_nodes:
        :return: list of rules
        """

        start_time = time.time()
        if self.display_logs:
            print("->Categorize patterns")

        # Checks all combinations found and checks for both 0 and 1 in the last pathology to study both cases.
        index = 0
        # TODO: PARALLEL
        while index < len(self.pattern_list_valid_nodes):
            for distinct_value in [self.target_true, self.target_false]:
                # UPDATE VALUES
                new_rule = copy.deepcopy(self.pattern_list_valid_nodes[index])
                new_rule.full_feature_comparer[-1].value = distinct_value

                number_negatives = self.count_query_negatives_query(test_data, new_rule.get_full_rule())
                number_positives = self.count_query_positives_query(test_data, new_rule.get_full_rule())

                #number_negatives = self.count_query_negatives(test_data, new_rule.full_feature_comparer)
                #number_positives = self.count_query_positives(test_data, new_rule.full_feature_comparer)

                number_positives_and_negatives = number_positives + number_negatives

                # If this rule has existing cases in total in the training set, is included.
                if number_positives_and_negatives > 0:
                    new_rule.number_all = number_positives_and_negatives
                    # Checks if the combinations show a rule for negative/positives
                    proportion_positives = number_positives / number_positives_and_negatives

                    # do not include rules with 0.5 prob
                    if proportion_positives == 0.5:
                        continue

                    if proportion_positives >= self.min_accuracy_coefficient:
                        # POSITIVES
                        new_rule.target_value = self.target_true
                        new_rule.number_target = number_positives
                        new_rule.target_accuracy = proportion_positives
                    else:
                        # NEGATIVES
                        proportion_negatives = number_negatives / number_positives_and_negatives
                        if proportion_negatives >= self.min_accuracy_coefficient:
                            new_rule.target_value = self.target_false
                            new_rule.number_target = number_negatives
                            new_rule.target_accuracy = proportion_negatives
                        else:
                            continue
                    self.rules_.append(new_rule)
            index += 1

        elapsed_time = time.time() - start_time
        if self.display_logs:
            print(f"Elapsed time to compute the categorize_patterns: {elapsed_time:.3f} seconds")

        return self.rules_

    def predict_unique_with_query_positives(self, pandas_dataset, feature_comparer):
        dataset_filtered = pandas_dataset
        for comparer in feature_comparer:
            dataset_filtered = comparer.unitary_loc(dataset_filtered)
        dataset_filtered = self.target_class_positive.unitary_loc(dataset_filtered)
        return dataset_filtered

    def predict_unique_with_query_positives_query(self, pandas_dataset, full_feature_comparer):
        dataset_filtered = pandas_dataset
        return chunk_query(dataset_filtered, concatenate_query(full_feature_comparer, self.target_class_positive.get_query()))

    def predict_unique_with_query_negatives(self, pandas_dataset, feature_comparer):
        dataset_filtered = pandas_dataset
        for comparer in feature_comparer:
            dataset_filtered = comparer.unitary_loc(dataset_filtered)
        dataset_filtered = self.target_class_negative.unitary_loc(dataset_filtered)
        return dataset_filtered

    def predict_unique_with_query_negatives_query(self, pandas_dataset, full_feature_comparer):
        dataset_filtered = pandas_dataset
        return chunk_query(dataset_filtered, concatenate_query(full_feature_comparer, self.target_class_negative.get_query()))

    def count_query_positives(self, pandas_dataset, feature_comparer):
        return len(self.predict_unique_with_query_positives(pandas_dataset, feature_comparer))

    def count_query_negatives(self, pandas_dataset, feature_comparer):
        return len(self.predict_unique_with_query_negatives(pandas_dataset, feature_comparer))

    def count_query_positives_query(self, pandas_dataset, full_query):
        return len(self.predict_unique_with_query_positives_query(pandas_dataset, full_query))

    def count_query_negatives_query(self, pandas_dataset, full_query):
        return len(self.predict_unique_with_query_negatives_query(pandas_dataset, full_query))

    def binary_tree_generator(self,
                              dataset,
                              node_value=0,
                              feature_index=0,
                              parent_node=None):
        """
        Función recursiva encargada de generar el árbol de nodos con sus respectivas queries y obtener en cada nodo la query y el número de fallecimientos y supervivencias de cada uno.

        :param dataset: Pandas DataFrame. Dataset con las filas para obtener el número de fallecimientos y defunciones usando cada query.
        :param node_value: Representa el valor de la característica en ese nodo en concreto.
        :param feature_index: índice auxiliar de la lista de características
        :param parent_node: node of the parent of current node
        :return:
        """
        current_feature_name = self.most_important_features_[feature_index]

        feature_comparer = FeatureComparer(current_feature_name, '==', node_value)
        # Caso base para el que se considera como nodo padre de todos.
        if parent_node is None:
            # Create Node
            current_node = Node(ID=0,
                                PARENT_ID=None,
                                number_positives=self.count_query_negatives_query(dataset, ''),
                                number_negatives=self.count_query_negatives_query(dataset, ''),
                                full_feature_comparer=[])
            # Incluye el nodo en la lista
            self.nodes_dict[current_node.ID] = current_node
            self.nodes_dict_ids.append(current_node.ID)

            # Una vez creado el padre, se accede a la primera característica, que representaría el primer nivel.
            # Por cada posible valor que pueda tomar esa característica, se crea un hijo nodo de manera recursiva
            for node_value in dataset[current_feature_name].unique():
                self.binary_tree_generator(dataset, node_value=node_value, parent_node=current_node)

        # Caso en el que el padre ya ha sido creado
        else:
            full_rule_query = concatenate_query(parent_node.get_full_query(), feature_comparer.get_query())
            number_negatives = self.count_query_negatives_query(dataset, full_rule_query)
            number_positives = self.count_query_positives_query(dataset, full_rule_query)

            full_comparer = parent_node.full_feature_comparer + [feature_comparer]
            #number_negatives = self.count_query_negatives(dataset, full_comparer)
            #number_positives = self.count_query_positives(dataset, full_comparer)

            node_values_total = number_negatives + number_positives
            # Si el nodo se considera que no tiene los casos suficientes,
            # es descartado y el árbol no continúa en esa rama.
            if node_values_total >= self.min_number_class_per_node:
                # Se le asigna la ID al nodo como la siguiente a la última utilizada.

                node_dict_ID = self.nodes_dict_ids[-1] + 1
                current_node = Node(ID=node_dict_ID,
                                    PARENT_ID=parent_node.ID,
                                    number_negatives=number_negatives,
                                    number_positives=number_positives,
                                    full_feature_comparer=full_comparer
                                    )

                # Incluye el nodo en la lista
                self.nodes_dict[current_node.ID] = current_node
                self.nodes_dict_ids.append(current_node.ID)

                # La ID del nodo es incluida en la lista de hijos del padre.
                self.nodes_dict[parent_node.ID].children.append(node_dict_ID)

                #new_dataset = dataset.loc[:, dataset.columns != current_feature_name]
                new_feature_index = feature_index + 1
                if new_feature_index >= len(self.most_important_features_):
                    # No hay más niveles
                    return
                # Por cada posible valor que pueda tomar esa nueva característica, se crea un hijo nodo de manera recursiva
                for node_value in dataset[self.most_important_features_[new_feature_index]].unique():
                    self.binary_tree_generator(dataset, node_value=node_value,
                                               feature_index=new_feature_index,
                                               parent_node=current_node)

    def generate_nodes(self, pandas_dataset, feature_importances):
        # List of top % important features in the model are obtained. This % regulated by coefficient between [0,1].
        self.get_top_important_features_list(feature_importances)

        # Generate Tree
        return self.generate_tree(pandas_dataset=pandas_dataset), self.most_important_features_

    def generate_tree(self, pandas_dataset):
        # Genera el árbol binario y obtiene las combinaciones que indican que hay un patrón:

        if self.most_important_features_ is None:
            return False

        minimal_dataset = copy.deepcopy(pandas_dataset[self.define_minimal_columns()])

        if self.display_logs:
            print("->Generate new tree based on list")
        start_time = time.time()
        self.binary_tree_generator(dataset=minimal_dataset)
        elapsed_time = time.time() - start_time
        if self.display_logs:
            print(f"Elapsed time to compute the binary_tree_generator: {elapsed_time:.3f} seconds")

        return self.nodes_dict

    def define_minimal_columns(self):
        return self.most_important_features_ + [self.target_value_name]

    def fit(self, pandas_dataset, feature_importances, node_dict=None, most_important_features=None):
        """
        Get list of top features and generate rules
        :param pandas_dataset:
        :return:
        @type pandas_dataset: pandas dataset
        @type node_dict: object
        @param node_dict:
        @param feature_importances:
        """

        if node_dict is not None:
            self.nodes_dict = node_dict

        if most_important_features is not None:
            self.most_important_features_ = most_important_features

        # if dict is null calculate it
        if not self.nodes_dict:
            self.generate_nodes(pandas_dataset, feature_importances)

        if not self.most_important_features_:
            self.get_top_important_features_list(feature_importances)

        # Lista de nodos válidos
        self.obtain_pattern_list_of_valid_nodes_with_pvalue()

        minimal_dataset = copy.deepcopy(pandas_dataset[self.define_minimal_columns()])
        # Categoriza patrones
        self.categorize_patterns(minimal_dataset)

        return self

    def sorting(self, sorting_method="target_accuracy"):
        match sorting_method:
            case "target_accuracy":
                return sorted(self.rules_, key=lambda r: r.target_accuracy, reverse=True)
            case "complexity":
                return sorted(self.rules_, key=lambda r: r.get_complexity(), reverse=True)
            case "p_value":
                return sorted(self.rules_, key=lambda r: r.p_value)
            case "chi2_statistic":
                return sorted(self.rules_, key=lambda r: r.chi2_statistic, reverse=True)
            case default:
                return sorted(self.rules_, key=lambda r: r.target_accuracy, reverse=True)

    def predict(self, X, sorting_method="target_accuracy"):
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
            @param X:
            @param sorting_method:
            @return:
        """
        predictions = []
        for x in X:
            categorize_rule = None
            for rule in self.sorting(sorting_method):
                prediction = rule.Predict(x)
                if prediction is not None:
                    predictions.append(prediction)
                    categorize_rule = True
                    break
            if categorize_rule is None:
                predictions.append(None)

        return predictions

    def description(self):
        display = '> ++++++++++++++++++++++++++++\n'
        display += f'> iPRules --  Number of Rules {len(self.rules_)}:\n'
        display += '> ++++++++++++++++++++++++++++\n'
        return display

    def __str__(self):
        display = self.description()
        sorted_rules = self.sorting()
        for num in range(len(sorted_rules)):
            display += f'{sorted_rules[num]}'

        return display
