import re

import numpy as np
from scipy.stats import chi2_contingency, chi2
from sklearn.base import ClassifierMixin
from sklearn.tree import BaseDecisionTree
from sklearn.tree._tree import Tree
from sklearn.utils.validation import check_is_fitted


class Node:
    def __init__(self, ID, PARENT_ID, deaths, survivors, chi_sq_death, chi_sq_surv, query):
        self.ID = ID  # int
        self.PARENT_ID = PARENT_ID  # int
        self.deaths = deaths  # int
        self.survivors = survivors  # int
        self.chi_sq_death = chi_sq_death  # float
        self.chi_sq_survival = chi_sq_surv  # float
        self.query = query  # str
        self.children = []  # list of int. Contiene todos los IDs de los hijos


def calc_chisq_node(parent_d, parent_s, deaths, survivors):
    """
    Calcula los coeficientes de chi-square usando los valores de muertes y
    supervivencias del nodo en cuestión y del nodo padre
    :return:
    :param parent_d:
    :param parent_s:
    :param deaths:
    :param survivors:
    :return:
    """
    epsilon_exp = 0.0001  # Coeficiente usado para evitar que la bondad de ajuste sea 0
    # Calculamos las bondades de ajuste
    expected_d = parent_d * (deaths + survivors) / (parent_d + parent_s)
    expected_s = parent_s * (deaths + survivors) / (parent_d + parent_s)
    if expected_d == 0:
        expected_d = epsilon_exp
    if expected_s == 0:
        expected_s = epsilon_exp
    # Los valores de chi-square
    chisq_d = ((deaths - expected_d) ** 2) / expected_d
    chisq_s = ((survivors - expected_s) ** 2) / expected_s

    return chisq_d, chisq_s


class iPRules(ClassifierMixin, BaseDecisionTree):

    def __init__(self,
                 base_ensemble,
                 feature_names,
                 target_value_name="target",
                 target_true="1",
                 target_false="0",
                 chi_square_probability=0.95,
                 scale_feature_coefficient=0.85
                 ):
        self.feature_importance_list = None
        self.most_important_features = None
        self.nodes = []
        self.base_ensemble = base_ensemble
        self.target_value_name = target_value_name
        self.target_true = target_true
        self.target_false = target_false
        self.death_query = f"{target_value_name} == {self.target_true}"
        self.surv_query = f"{target_value_name} == {self.target_false}"
        self.chi_square_probability = chi_square_probability
        self.scale_feature_coefficient = scale_feature_coefficient
        self.feature_names = feature_names

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
        return self.nodes[index].deaths, self.nodes[index].survivors

    # Calcula los coeficientes de chi-square usando los valores de muertes y
    # supervivencias del nodo en cuestión y del nodo padre

    def has_pattern(self, matrix):
        matrix[matrix == 0] = 0.0001
        # print('Matrix is:',matrix)
        stat, p, dof, expected = chi2_contingency(matrix, correction=False)
        critical = chi2.ppf(self.chi_square_probability, dof)
        if abs(stat) >= critical:
            return True
        else:
            return False

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

    def obtain_patterns(self):
        """

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
                            aux_matrix.append([aux_node.survivors, aux_node.deaths])
                            # Se parsea la query para representar el nivel.
                            aux_query = re.search("(.*) ==.*", aux_node.query)
                        # if has_enough_cases:  # Se calcula el p valor de los hermanos en ese subnivel
                        np_matrix = np.array(aux_matrix).astype(float)
                        np_matrix = np_matrix.transpose()
                        has_pattern = self.has_pattern(np_matrix)
                        if has_pattern:
                            patterns.add(
                                aux_query[1])  # Si se encuentra una regla que puede tener un patrón, se incluye.
        return patterns

    def categorize_patterns(self, test_data, patterns, coefficient):
        """

        :param test_data:
        :param patterns:
        :param coefficient:
        :return:
        """
        death_patterns = []
        surv_patterns = []
        for i in range(0,
                       len(patterns)):  # Checks all combinations found and checks for both 0 and 1 in the last pathology to study both cases.
            values = [0, 1]
            for j in values:
                query = patterns[i] + ' == ' + str(j)  # Adds the 0 or 1 to the last pathology
                deaths = len(test_data.query(query + f" & {self.target_value_name} == {self.target_true}"))
                survs = len(test_data.query(query + f" & {self.target_value_name} == {self.target_false}"))
                if survs + deaths > 0:  # If this pattern has existing cases in total in the training set, is included.
                    if (deaths / (
                            survs + deaths)) >= coefficient:  # Checks if the combinations show a pattern for death/surv
                        death_patterns.append([patterns[i] + ' == ' + str(j), deaths, survs + deaths])
                    elif (survs / (survs + deaths)) >= coefficient:
                        surv_patterns.append([patterns[i] + ' == ' + str(j), survs, survs + deaths])
        return death_patterns, surv_patterns

    def binary_tree_generator(self, dataset, query='', node_value=0, feature_index=0, parent_id=-1):
        """
        Función recursiva encargada de generar el árbol de nodos con sus respectivas queries y obtener en cada nodo la query y el número de fallecimientos y supervivencias de cada uno.

        :param features: list of str. Contiene todas las características anteriormente seleccionadas
        :param dataset: Pandas DataFrame. Dataset con las filas para obtener el número de fallecimientos y defunciones usando cada query.
        :param query: Variable auxiliar en la que se irá anexando las características junto con sus valores para generar las queries.
        :param node_value: Representa el valor de la característica en ese nodo en concreto.
        :param feature_index: índice auxiliar de la lista de características
        :param parent_id:
        :return:
        """
        # En el caso de que queden características para ampliar el nivel:
        if feature_index < len(self.most_important_features):

            # Caso base para el que se considera como nodo padre de todos.
            if parent_id == -1:
                self.nodes.append(Node(ID=0,
                                       PARENT_ID=None,
                                       deaths=len(dataset.query(self.death_query)),
                                       # Guarda la cantidad de muertes totales.
                                       survivors=len(dataset.query(self.surv_query)),
                                       # Guarda la cantidad de supervivientes totales.
                                       chi_sq_death=0,  # El padre no tiene chi_sq, osea que se deja por defecto a 0.
                                       chi_sq_surv=0,
                                       query=None,  # El padre tampoco tiene query, osea que se deja a 0.
                                       ))
                # Una vez creado el padre, se accede a la primera característica, que representaría el primer nivel.
                # Por cada posible valor que pueda tomar esa característica, se crea un hijo nodo de manera recursiva
                for i in dataset[self.most_important_features[0]].unique():
                    self.binary_tree_generator(dataset, parent_id=0, node_value=i)

            # Caso en el que el padre ya ha sido creado
            else:
                # A la query auxiliar se le incluye la característica del nodo junto con el valor asignado
                # tal que queda como FEATURE == X
                aux_query = self.most_important_features[feature_index] + ' == ' + str(node_value)

                # Para poder anexar las características, hay que incluir '&' entre cada valor, pero se puede dar el caso
                # en el que la query esté vacía y poner un '&' no funcionaría, por lo que hay 2 casos posibles:
                # -Query inicial(Primer nivel de hijos).
                # -Subniveles con queries no nulas.
                if query != '':
                    aux_query = query + " & " + aux_query
                else:
                    aux_query = query + aux_query

                # Se obtienen cantidad de supervivencias y defunciones de ese nodo en concreto ejecutando las queries en el dataset.
                deaths = len(dataset.query(self.death_query + ' & ' + aux_query))
                survivors = len(dataset.query(self.surv_query + ' & ' + aux_query))
                # Los valores de muertes y supervivencias del padre se obtienen para calcular el chi-square
                parent_d, parent_s = self.get_ds_node(parent_id)
                aux_d, aux_s = calc_chisq_node(parent_d, parent_s, deaths, survivors)

                # Si el nodo se considera que no tiene los casos suficientes, es descartado y el árbol no continúa en esa rama.
                if (deaths + survivors) >= 3:
                    # Se le asigna la ID al nodo como la siguiente a la última utilizada.
                    node_ID = self.nodes[-1].ID + 1
                    self.nodes.append(Node(ID=node_ID,
                                           PARENT_ID=parent_id,
                                           deaths=deaths,
                                           survivors=survivors,
                                           chi_sq_death=aux_d,
                                           chi_sq_surv=aux_s,
                                           query=aux_query,
                                           ))

                    self.add_child_to_parent(parent_id,
                                             node_ID)  # La ID del nodo es incluida en la lista de hijos del padre.

                    new_parent_id = self.nodes[-1].ID  # Este nodo ahora hará de padre.
                    for i in dataset[self.most_important_features[feature_index]].unique():
                        new_feature_index = feature_index + 1
                        self.binary_tree_generator(dataset, query=aux_query, node_value=i,
                                                   feature_index=new_feature_index, parent_id=new_parent_id)

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
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba

    def __str__(self):
        s = '> ------------------------------\n'
        s += '> iPRules:\n'
        s += '> ------------------------------\n'
        return s + str(list(self.obtain_patterns())) + '\n'
