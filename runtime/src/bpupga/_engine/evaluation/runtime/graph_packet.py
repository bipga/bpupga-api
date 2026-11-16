"""Serialize topology and native iteration order, never node attributes/labels."""
import networkx as nx
import numpy as np


def save_graphs(path, graphs):
    values = {}
    for number,graph in enumerate(graphs,1):
        neighbors = [list(graph.neighbors(u)) for u in range(len(graph))]
        values.update({f"n{number}":len(graph),
                       f"edges{number}":np.asarray(list(graph.edges()),dtype=np.int64).reshape(-1,2),
                       f"indptr{number}":np.r_[0,np.cumsum([len(x) for x in neighbors])],
                       f"neighbors{number}":np.asarray([v for row in neighbors for v in row],dtype=np.int64)})
    np.savez(path,**values)


def load_graphs(path):
    graphs=[]
    with np.load(path,allow_pickle=False) as arrays:
        for number in (1,2):
            graph=nx.Graph()
            graph.add_nodes_from(range(int(arrays[f"n{number}"])))
            graph.add_edges_from(arrays[f"edges{number}"].tolist())
            indptr,neighbors=arrays[f"indptr{number}"],arrays[f"neighbors{number}"]
            for u in graph:
                order=neighbors[indptr[u]:indptr[u+1]].tolist()
                if set(order)!=set(graph.neighbors(u)):
                    raise ValueError("Malformed adjacency-order packet")
                # Preserve original adjacency iteration order, which matters to
                # seeded random-walk methods. Shared edge attribute dicts stay
                # the same objects on the two sides of an undirected edge.
                graph._adj[u]={v:graph._adj[u][v] for v in order}
            graphs.append(graph)
    return graphs
