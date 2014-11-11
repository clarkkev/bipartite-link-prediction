import util
import numpy as np
from scipy import sparse
import networkx as nx


def main():
    print "Loading data and building transition matrix..."
    examples = util.load_json('./data/train/examples.json')
    G = nx.read_edgelist('./data/train/graph.txt', nodetype=int)
    adjacency_matrix = nx.adjacency_matrix(G)
    inverse_degree_matrix = sparse.diags([[1.0 / adjacency_matrix.getrow(i).sum()
                                           for i in range(adjacency_matrix.shape[0])]], [0])
    transition_matrix = inverse_degree_matrix.dot(adjacency_matrix)

    print "Running random walks..."
    for u in util.logged_loop(examples, util.LoopLogger(10, len(examples), True)):
        p = run_random_walk(transition_matrix, int(u), 10).todense()
        for b in examples[u]:
            examples[u][b] = p[0, int(b)]

    util.write_json(examples, './data/train/random_walks.json')


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
