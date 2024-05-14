import os
import kahypar as kahypar

num_nodes = 7
num_nets = 6

hyperedge_indices = [0, 2, 4, 6, 8, 10, 12]
hyperedges = [0, 2, 0, 1, 3, 4, 3, 5, 6, 2, 5, 6]

node_weights = [1, 2, 3, 4, 5, 6, 9]
edge_weights = [11, 22, 33, 44, 55, 66]

k = 2

hypergraph = kahypar.Hypergraph(num_nodes, num_nets, hyperedge_indices, hyperedges, k, edge_weights, node_weights)

context = kahypar.Context()
context.loadINIconfiguration("../km1_kKaHyPar_sea20.ini")

context.setK(k)
context.setEpsilon(0.03)

kahypar.partition(hypergraph, context)

nodes_blocks_index_list = [[] for i in range(k)]

edges_blocks_index_list = [[] for j in range(k)]

# 获取每一个节点所在的区块ID
for p in hypergraph.nodes():
    nodes_blocks_index_list[hypergraph.blockID(p)].append(p)

for start in range(0, len(hyperedges), 2):
    for i, array in enumerate(nodes_blocks_index_list):
        if hyperedges[start] in array and hyperedges[start + 1] in array:
            edges_blocks_index_list[i].append((hyperedges[start], hyperedges[start + 1]))

print(nodes_blocks_index_list)
print(edges_blocks_index_list)
