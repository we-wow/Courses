"""

@Time   : 2021/9/30 12:27
@Author : Wei Mingjiang
@File   : input_data_preprocess.py
@Version: 1.0.0
@Content: 
"""
import csv
import numpy as np
import itertools


class Graph:
    def __init__(self, topology_file_path: str):
        with open(topology_file_path) as f:
            # Read topology of a graph.
            reader = csv.reader(f)
            # Convert indexes, which start from 1 in graph but start from 1 in python's iterable objects.
            unsorted_edge = np.array([[int(_) - 1 for _ in edge] for edge in reader])
            self.edges = unsorted_edge[np.lexsort((unsorted_edge[:, 0],))]
        # Number of edges and vertexes
        self.edges_number = self.edges.shape[0]
        self.vertex_number = int(np.max(self.edges[:, 1:3]) + 1)
        # Generate a incidence matrix
        self.incidence_matrix = self.generate_incidence_matrix()

    def generate_incidence_matrix(self):
        """
        Generate the incidence matrix of the Graph.
        :return:
        """
        incidence_matrix = np.zeros(shape=(self.vertex_number, self.edges_number))
        for edge in self.edges:
            incidence_matrix[edge[1], edge[0]] = 1
            incidence_matrix[edge[2], edge[0]] = -1
        return incidence_matrix

    def find_all_spanning_tree_by_iteration(self):
        """
        Find all spanning trees by iterative method.
        First generate all possible sets of edges, and then, verify whether it is a "spanning tree"
        by calculate the determinant of the corresponding sub-matrix.
        If the determinant is not zero, the corresponding set of edges is a "spanning tree".
        :return: A list[set] which contains all spanning trees.
        """
        spanning_trees = []
        for cols in itertools.combinations(range(self.edges_number), self.vertex_number - 1):
            # If the determinant of the selected columns is not zero, then set of the corresponding edges
            # can be a spanning tree
            if np.linalg.det(self.incidence_matrix[:self.vertex_number - 1, :].take(cols, 1)) != 0:
                spanning_trees.append({edge + 1 for edge in cols})
        return spanning_trees

    def generate_fundamental_cut_set_matrix(self, tree=None):
        if tree is None:
            # If there isn't a given tree, then we need to find a tree first.
            for cols in itertools.combinations(range(self.edges_number), self.vertex_number - 1):
                # If the determinant of the selected columns is not zero, then set of the corresponding edges
                # can be a spanning tree
                if np.linalg.det(self.incidence_matrix[:self.vertex_number - 1, :].take(cols, 1)) != 0:
                    tree = {_ for _ in cols}
                    break
        else:
            # indexes conversion
            tree = {_ - 1 for _ in tree}
        # Initialize a fundamental_cut_set_matrix
        fundamental_cut_set_matrix = np.zeros(shape=(self.vertex_number - 1, self.edges_number))
        # Initialize tree branches in matrix
        for i, t in zip(range(self.vertex_number - 1), sorted(tree)):
            fundamental_cut_set_matrix[i, t] = 1
        # Find the cut set of each tree branch
        for i, branch in zip(range(self.vertex_number - 1), tree):
            # Search vertexes of each fundamental cut sub-set

            # Copy tree
            tree_cut_one_branch = {_ for _ in tree}
            # Cut this tree branch, and form two spanning sub-trees
            tree_cut_one_branch.remove(branch)
            sub_tree_out = set()
            sub_tree_in = set()
            sub_tree_out.add(self.edges[branch, 1])
            sub_tree_in.add(self.edges[branch, 2])
            # Search all vertexes of each spanning sub-tree based on BFS
            while True:
                set_length = len(sub_tree_out)
                tree_cut_one_branch_for_iteration = {_ for _ in tree_cut_one_branch}
                for t in tree_cut_one_branch_for_iteration:
                    if self.edges[t, 1] in sub_tree_out:
                        sub_tree_out.add(self.edges[t, 2])
                    elif self.edges[t, 2] in sub_tree_out:
                        sub_tree_out.add(self.edges[t, 1])
                    else:
                        continue
                    tree_cut_one_branch.remove(t)
                if len(sub_tree_out) == set_length:
                    break
            while True:
                set_length = len(sub_tree_in)
                tree_cut_one_branch_for_iteration = {_ for _ in tree_cut_one_branch}
                for t in tree_cut_one_branch_for_iteration:
                    if self.edges[t, 1] in sub_tree_in:
                        sub_tree_in.add(self.edges[t, 2])
                    elif self.edges[t, 2] in sub_tree_in:
                        sub_tree_in.add(self.edges[t, 1])
                    else:
                        continue
                    tree_cut_one_branch.remove(t)
                if len(sub_tree_in) == set_length:
                    break
            link_branch = {_ for _ in range(self.edges_number) if _ not in tree}
            for l in link_branch:
                if self.edges[l, 1] in sub_tree_out and self.edges[l, 2] in sub_tree_in:
                    fundamental_cut_set_matrix[i, l] = 1
                elif self.edges[l, 1] in sub_tree_in and self.edges[l, 2] in sub_tree_out:
                    fundamental_cut_set_matrix[i, l] = -1
        return fundamental_cut_set_matrix, tree

    def find_all_spanning_tree_by_polynomial(self, initial_tree=None):
        if initial_tree is not None and np.linalg.det(self.incidence_matrix[:self.vertex_number - 1,
                                                      :].take([_-1 for _ in initial_tree], 1)) == 0:
            raise ValueError(f'Initial set {initial_tree} is not a spanning tree')
        fundamental_cut_set_matrix, tree = self.generate_fundamental_cut_set_matrix(tree=initial_tree)
        cut_set_polynomials = []
        # Form the polynomials
        for i, t in zip(range(fundamental_cut_set_matrix.shape[0]), sorted(tree)):
            cut_set_polynomials.append([_ for _ in range(fundamental_cut_set_matrix.shape[1])
                                        if fundamental_cut_set_matrix[i][_] != 0])
        # Multiply polynomials
        base_polynomial = [[_, ] for _ in cut_set_polynomials[0]]
        for p in cut_set_polynomials[1:]:
            temp_base_polynomial = []
            # Get tuples of cartesian product of items for multiplying
            for elements in itertools.product(base_polynomial, p):
                if elements[1] not in elements[0]:
                    temp_base_polynomial.append([_ for _ in elements[0]] + [elements[1], ])
            # Filter same items
            for elements in itertools.combinations(temp_base_polynomial, 2):
                if set(elements[0]) == set(elements[1]):
                    try:
                        temp_base_polynomial.remove(elements[0])
                        temp_base_polynomial.remove(elements[1])
                    except ValueError:
                        pass
            base_polynomial = [_ for _ in temp_base_polynomial]
        return [{__+1 for __ in _} for _ in base_polynomial]


if __name__ == '__main__':
    g = Graph(topology_file_path='../data/example2.csv')
    all_tree_1 = g.find_all_spanning_tree_by_polynomial(initial_tree={1, 4, 5})
    all_tree_2 = g.find_all_spanning_tree_by_iteration()
