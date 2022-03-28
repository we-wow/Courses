"""

@Author : Wei Mingjiang
@Time   : 2022/3/24 13:46
@File   : graph.py
@Version: 0.1.0
@Content: First version.
"""
import queue
from collections import deque

import numpy as np


class Node:
    def __init__(self, name, no: int):
        self.name = name
        self.no = no


class Edge:
    def __init__(self, name, u: int, v: int, ):
        self.name = name
        self.u = u
        self.v = v


class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        with open("data/node", "r") as f:
            for line in f.readlines():
                if not line:
                    continue
                data = line.strip().split()
                name, no = data
                self.nodes.append(Node(name, int(no) - 1))
        self.num_node = len(self.nodes)
        self.adjacency_matrix = np.zeros((self.num_node, self.num_node))
        with open("data/edge", "r") as f:
            for line in f.readlines():
                if not line:
                    continue
                data = line.strip().split()
                name, u, v = data
                i, j = int(u) - 1, int(v) - 1
                self.edges.append(Edge(name, i, j))
                self.adjacency_matrix[i][j] = 1
                self.adjacency_matrix[j][i] = 1

    def breath_first_search(self, source=25):
        que = queue.Queue()
        que.put(source)
        visited = [False for _ in range(self.num_node)]
        visited[source] = True
        while not que.empty():
            visiting = que.get()
            print(f"{visiting + 1} -> ", end="")
            for i in range(self.num_node):
                n = self.adjacency_matrix[visiting, i]
                if n and not visited[i]:
                    que.put(i)
                    visited[i] = True
        print()

    def depth_first_search(self, source=0):
        dq = deque()
        dq.append(source)
        visited = [False for _ in range(self.num_node)]
        visited[source] = True
        while len(dq):
            visiting = dq.pop()
            print(f"{visiting + 1} -> ", end="")
            for i in range(self.num_node):
                n = self.adjacency_matrix[visiting, i]
                if n and not visited[i]:
                    dq.append(i)
                    visited[i] = True
        print()


if __name__ == '__main__':
    g = Graph()
    g.breath_first_search()
    g.depth_first_search()
