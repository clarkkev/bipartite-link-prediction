import util
import datetime
import networkx as nx
import numpy as np
from scipy import sparse
from dataset_maker import get_date


def main(weight_edges=False):
    print "Loading data and building transition matrix..."
    examples = util.load_json('./data/train/examples.json')
    G = nx.read_edgelist('./data/train/graph.txt', nodetype=int)
    if weight_edges:
        reviews = util.load_json('./data/train/review.json')
        end_date = datetime.date(2013, 1, 1)
        edges = G.edges()
        for e in util.logged_loop(edges, util.LoopLogger(20000, len(edges), True)):
            n1, n2 = str(e[0]), str(e[1])
            if n1 not in reviews or n2 not in reviews[n1]:
                n1, n2 = n2, n1
            G[e[0]][e[1]]['weight'] = 1.0 / ((end_date - get_date(reviews[n1][n2][0])).days + 90)
        del reviews  # save some memory

    adjacency_matrix = nx.adjacency_matrix(G)
    inverse_degree_matrix = sparse.diags([[1.0 / adjacency_matrix.getrow(i).sum()
                                           for i in range(adjacency_matrix.shape[0])]], [0])
    transition_matrix = inverse_degree_matrix.dot(adjacency_matrix)

    print "Running random walks..."
    for u in util.logged_loop(examples, util.LoopLogger(10, len(examples), True)):
        p = run_random_walk(transition_matrix, int(u), 10).todense()
        for b in examples[u]:
            examples[u][b] = p[0, int(b)]

    util.write_json(examples, './data/train/' + ('weighted_random_walks.json' if weight_edges
                                                 else 'random_walks.json'))


def run_random_walk(transition_matrix, u, iterations=10, jump_p=0.2):
    p = np.zeros(transition_matrix.shape[0])
    p[u] = 1.0
    p = sparse.csr_matrix(p)

    # solving for the stationary distribution exactly is not feasible, so we instead just run a
    # random walk for some number of iterations
    for i in range(iterations):
        p = np.dot(p, transition_matrix)
        p *= (1 - jump_p)
        p[0, u] += jump_p

    return p


if __name__ == '__main__':
    main()
