import dgl
import numpy as np
import torch

g = dgl.graph(([0,0,0,0,0], [1,2,3,4,5]), num_nodes=6)

print(g.edges())

# g.ndata['x'] = torch.randn(6,3)

# g.edata['a'] = torch.randn(5,4)

# g.ndata['y'] = torch.randn(6,5,4)

# print(g.edata['a'])
# print(f"Number of nodes: {g.num_nodes()}")
# print(f"Number of edges: {g.num_edges()}")
# print(f"Number of out degree of center edges: {g.out_degrees(0)}")
# print(f"Number of in degree of center edges: {g.in_degrees(0)}")

# sg1 = g.subgraph([0,1,3])
# sg2 = g.edge_subgraph([0,1,3])
# print(sg1.ndata[dgl.NID])
# print(sg1.edata[dgl.EID])
# print(sg2.ndata[dgl.NID])
# print(sg2.edata[dgl.EID])

# print(sg1.ndata['x'])
# print(sg1.edata['a'])
# print(sg2.ndata['x'])
# print(sg2.edata['a'])

# newg = dgl.add_reverse_edges(g)
# newg.edges()

# dgl.save_graphs('graph.dgl', g)
# dgl.save_graphs('graphs.dgl', [g, sg1, sg2])

# (g,),_ = dgl.load_graphs('graph.dgl')
# print(g)
# (g,sg1,sg2),_ = dgl.load_graphs('graphs.dgl')
# print(g)
# print(sg1)
# print(sg2)