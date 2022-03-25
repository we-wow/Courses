from utils.graph import Graph

g = Graph(topology_file_path='data/example2.csv')
# If there isn't a given initial tree, program will find one automatically
# all_tree_1 = g.find_all_spanning_tree_by_polynomial()
all_tree_1 = g.find_all_spanning_tree_by_polynomial(initial_tree={1, 2, 5})
all_tree_2 = g.find_all_spanning_tree_by_iteration()
print(all_tree_1)
print(all_tree_2)