import networkx as nx


T = nx.random_tree(n=15)

nx.write_adjlist(T,"adj_list_test")
