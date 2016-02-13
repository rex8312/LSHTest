__author__ = 'rex8312'

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
from sklearn.metrics import euclidean_distances
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn import cross_validation
from sklearn import svm


class Node(object):
    def __init__(self, parent=None, max_depth=32):
        self.parent = parent
        self.children = list()
        self.idx = list()
        self.table = dict()
        self.max_depth = max_depth

    def add(self, binary, idx):
        self.idx.append(idx)

        if len(binary) > 0:
            self.children.extend([Node(self, self.max_depth-1), Node(self, self.max_depth-1)])
            self.children[binary[0]].add(binary[1:], idx)

    def find_prefix_match(self, hashed_query):
        current = self
        k = 0
        for b in hashed_query:
            if len(current.children) > 0:
                current = current.children[b]
                k += 1
            else:
                break
        return k

    def query(self, binary, max_depth):
        current_node = self
        current_depth = 0
        for b in binary:
            if len(current_node.children) > 0 and current_depth < max_depth:
                current_node = current_node.children[b]
            else:
                return list()
            current_depth += 1
        return current_node.idx

    def make_table(self):
        cs = list()

        def gen_cs(current):
            if len(current) == self.max_depth:
                return
            else:
                add0 = current[:]
                add0.append(0)
                add1 = current[:]
                add1.append(1)
                cs.extend([add0, add1])
                gen_cs(add0)
                gen_cs(add1)

        gen_cs(list())

        for case in sorted(cs):
            self.table[''.join(['1' if x == 1 else '0' for x in case])] = self.query(case, self.max_depth)


class LSH_forest(BaseEstimator, ClassifierMixin):
    def __init__(self, max_label_length=32, number_of_trees=5, c=1, m=10):
        self.debug = False
        self.max_label_length = max_label_length
        self.number_of_trees = number_of_trees
        self.min_label_length = 20
        if self.debug:
            self.random = np.random.RandomState(seed=1)
        else:
            self.random = np.random.RandomState()
        self.c = c
        self.m = m

    def _get_random_hyperplanes(self, hash_size, dim):
        return self.random.randn(hash_size, dim)

    def _hash(self, x, hash_function):
        projection = np.dot(hash_function, x)
        return [1 if v > 0 else 0 for v in projection]

    def _create_tree(self, hash_function):
        number_of_points = self.xs.shape[0]
        root = Node(max_depth=self.max_label_length)
        for i in range(number_of_points):
            binary = self._hash(self.xs[i], hash_function)
            root.add(binary, i)
        return root

    def build_index(self):
        dim = self.xs.shape[1]

        self.hash_functions = list()
        self.trees = list()
        self.original_indices = list()

        for i in range(self.number_of_trees):
            hash_size = self.max_label_length
            hash_function = self._get_random_hyperplanes(hash_size, dim)
            tree = self._create_tree(hash_function)
            if self.debug: tree.make_table()
            self.trees.append(tree)
            self.hash_functions.append(hash_function)

    def query(self, query):
        c = self.c
        m = self.m
        query = np.array(query)

        # descend phase
        max_depth = 0
        for i in range(len(self.trees)):
            bin_query = self._hash(query, self.hash_functions[i])
            k = self.trees[i].find_prefix_match(bin_query)
            if k > max_depth:
                max_depth = k

        # asynchronous ascend phase
        candidates = list()
        number_of_candidates = c * len(self.trees)
        while max_depth > 0 and (len(candidates) < number_of_candidates or len(set(candidates)) < m):
            for i in range(len(self.trees)):
                bin_query = self._hash(query, self.hash_functions[i])
                candidates.extend(self.trees[i].query(bin_query, max_depth))
            max_depth = max_depth - 1

        if len(candidates) == 0:
            candidates = range(len(self.xs))

        candidates = np.array(list(set(candidates)))
        if self.debug:
            print('md:', max_depth)
            print('c:', candidates)
        distances = euclidean_distances(query, self.xs[candidates])
        return sorted(zip(distances[0], candidates))[:self.m]

    def fit(self, X, y):
        self.xs_max = np.max(X, axis=0)
        self.xs_min =  np.min(X, axis=0)
        self.xs_mean = np.mean(X, axis=0)
        self.xs_std = np.std(X, axis=0)
        self.xs = list()
        for _x in X:
            self.xs.append((_x - self.xs_min) / (self.xs_max - self.xs_min) * 2. - 1.)
        self.xs = np.array(self.xs)
        self.build_index()
        self.classes_, self.indices = np.unique(y, return_inverse=True)
        #print self.classes_, self.indices
        return self

    def predict(self, X):
        ys = list()
        for x in X:
            x = (x - self.xs_min) / (self.xs_max - self.xs_min) * 2. - 1.
            candidates = self.query(x)
            counts = np.zeros(len(self.classes_))
            for candidate in candidates:
                counts[self.indices[candidate[1]]] += 1
            ys.append(self.classes_[np.argmax(counts)])
        return ys


if __name__ == '__main__':
    """
    # 1
    xs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    ys = np.array([0, 1, 1, 1])
    clf = LSH_forest(max_label_length=3)
    clf.fit(xs, ys)
    print clf.predict([[0, 0], [0, 1], [1, 0], [1, 1]])
    """
    """
    # 2
    xs = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    ys = np.array([1, 1, 1, 2, 2, 2])
    clf = LSH_forest(max_label_length=3)
    clf.fit(xs, ys)
    print clf.predict([[-0.8, -1]])
    """
    """
    # 3
    iris = load_iris()
    xs = normalize(iris.data)
    ys = iris.target
    clf = LSH_forest(max_label_length=5)
    clf.fit(xs, ys)
    print clf.predict([[5.1, 3.5, 1.4, 0.2]])
    """
    # 4
    iris = load_iris()
    data = iris.data
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, iris.target, test_size=0.4)
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    clf = clf = LSH_forest(max_label_length=15, number_of_trees=10, c=1, m=5).fit(X_train, y_train)
    print(clf.score(X_test, y_test))