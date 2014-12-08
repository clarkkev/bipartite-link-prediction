import util
import networkx as nx
import numpy as np
from scipy import sparse


def svd_user_business(k=50):
    print "Loading data and building user-business matrix..."
    users = util.load_json('./data/test/user.json').keys()
    businesses = util.load_json('./data/test/business.json').keys()
    examples = util.load_json('./data/test/examples.json')

    user_to_row = dict(zip(users, range(len(users))))
    business_to_column = dict(zip(businesses, range(len(businesses))))

    user_business_matrix = sparse.lil_matrix((len(users), len(businesses)), dtype=float)
    with open('./data/test/graph.txt') as f:
        for line in f:
            u, b = line.split()
            user_business_matrix[user_to_row[u], business_to_column[b]] = 1
    user_business_matrix = sparse.csr_matrix(user_business_matrix)

    print "Computing singular value decomposition..."
    u, s, vt = sparse.linalg.svds(user_business_matrix, k=k)
    us = u * s

    print "Writing results..."
    for u in examples:
        for b in examples[u]:
            examples[u][b] = np.dot(us[user_to_row[u], :], vt[:, business_to_column[b]])
    util.write_json(examples, './data/test/' + 'svd_user_business.json')


def svd(k=50):
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
    svd()
