from utils.graph import Graph

g = Graph(topology_file_path='data/example2.csv')
all_tree_1 = g.find_all_spanning_tree_by_polynomial(initial_tree={1, 4, 5})
all_tree_2 = g.find_all_spanning_tree_by_iteration()
print(all_tree_1)
print(all_tree_2)