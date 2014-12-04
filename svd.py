import util
import networkx as nx
import numpy as np
from scipy import sparse


def main(k=50):
    print "Loading data and building adjacency matrix..."
    examples = util.load_json('./data/test/examples.json')
    G = nx.read_edgelist('./data/test/graph.txt', nodetype=int)
    adjacency_matrix = sparse.csr_matrix(nx.adjacency_matrix(G), dtype=float)

    print "Computing singular value decomposition..."
    u, s, vt = sparse.linalg.svds(adjacency_matrix, k=k)
    us = u * s

    print "Writing results..."
    for u in examples:
        for b in examples[u]:
            examples[u][b] = np.dot(us[u, :], vt[:, b])
    util.write_json(examples, './data/test/' + 'svd.json')


if __name__ == '__main__':
    main()
