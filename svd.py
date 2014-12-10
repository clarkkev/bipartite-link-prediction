import util
import networkx as nx
import numpy as np
from scipy import sparse


def svd_user_business(data_dir, k=50):
    print "Loading data and building user-business matrix..."
    users = util.load_json('./data/' + data_dir + '/user.json').keys()
    businesses = util.load_json('./data/' + data_dir + '/business.json').keys()
    examples = util.load_json('./data/' + data_dir + '/examples.json')

    user_to_row = dict(zip(users, range(len(users))))
    business_to_column = dict(zip(businesses, range(len(businesses))))

    user_business_matrix = sparse.lil_matrix((len(users), len(businesses)), dtype=float)
    with open('./data/' + data_dir + '/graph.txt') as f:
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
    util.write_json(examples, './data/' + data_dir + '/svd.json')


def svd(data_dir, k=50):
    print "Loading data and building adjacency matrix..."
    examples = util.load_json('./data/' + data_dir + '/examples.json')
    G = nx.read_edgelist('./data/' + data_dir + '/graph.txt', nodetype=int)
    adjacency_matrix = sparse.csr_matrix(nx.adjacency_matrix(G), dtype=float)

    print "Computing singular value decomposition..."
    u, s, vt = sparse.linalg.svds(adjacency_matrix, k=k)
    us = u * s

    print "Writing results..."
    for u in examples:
        for b in examples[u]:
            examples[u][b] = np.dot(us[u, :], vt[:, b])
    util.write_json(examples, './data/' + data_dir + '/svd.json')


if __name__ == '__main__':
    svd_user_business('train')
    svd_user_business('test')
