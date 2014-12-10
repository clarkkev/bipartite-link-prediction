from scipy import sparse
from dataset_maker import get_date
import numpy as np
import datetime
import util
import networkx as nx
import random
from collections import defaultdict

ALPHA = 0.3
H = 0.02
WMV_LOSS_WIDTH = 0.0005
REGULARIZATION_CONSTANT = 0.00001
LEARNING_RATE = 30
NUM_TRAIN_USERS = 400
MAX_POSITIVE_EDGES_PER_USER = 5
MAX_NEGATIVE_EDGES_PER_USER = 100
INITIAL_WEIGHTS = {
    "age": 2,
    "age_0.5": 0,
    "age_0.2": 0,
    "stars": 0,
    "liked": 0,
    "bias": -2
}


def f(x):
    return 1 / (1 + np.exp(-x))


def h(x):
    return 1 / (1 + np.exp(x / WMV_LOSS_WIDTH))


def get_features(reviews, is_train):
    end_date = datetime.date(2012, 1, 1) if is_train else datetime.date(2013, 1, 1)
    # we multiply some values by constants as a hacky way of normalizing the features
    return {
        "age": 50.0 / ((end_date - get_date(reviews[0])).days + 30),
        "age_0.5": 10.0 / (((end_date - get_date(reviews[0])).days + 30) ** 0.5),
        "age_0.2": 3.0 / (((end_date - get_date(reviews[0])).days + 30) ** 0.2),
        "stars": int(reviews[0]["stars"]) / 5.0,
        "liked": 1 if int(reviews[0]["stars"]) > 3 else 0,
        "bias": 1.0
    }


def get_phi(is_train):
    data_dir = 'train' if is_train else 'test'

    print "Loading reviews..."
    reviews = util.load_json('./data/' + data_dir + '/review.json')

    print "Building graph..."
    G = nx.read_edgelist('./data/' + data_dir + '/graph.txt', nodetype=int)
    n = G.number_of_nodes()

    print "Building feature matrices..."
    phi = defaultdict(lambda: sparse.lil_matrix((n, n), dtype=float))
    for (u, v) in G.edges():
        if str(u) not in reviews:
            u, v = v, u
        features = get_features(reviews[str(u)][str(v)], is_train)
        for feature_name, value in features.iteritems():
            phi[feature_name][u, v] = value
            phi[feature_name][v, u] = value

    print "Converting..."
    for k, m in phi.items():
        phi[k] = sparse.csr_matrix(m)

    return phi


def get_Q(phi, w):
    a = sparse.csr_matrix((phi[0].shape), dtype=float)
    for k in w:
        a = a + phi[k] * w[k]
    a.data = f(a.data)
    d_inv = sparse.diags([[1.0 / a.getrow(i).sum() for i in range(a.shape[0])]], [0])
    return d_inv.dot(a)


def get_ps(Q, old_ps, max_iter=50, convergence_criteria=1e-4, log=False, examples=None):
    ps = {}
    total_iterations = 0
    ll = util.LoopLogger(10, len(old_ps), True)
    for u in old_ps:
        if log:
            ll.step()
        ps[u], iterations = stationary_distribution(Q, u, old_ps[u], max_iter, convergence_criteria)
        if examples:
            for b in examples[str(u)]:
                examples[str(u)][b] = ps[u][0, int(b)]
            del ps[u]
        total_iterations += iterations
    print "  average_iterations {:.2f}".format(total_iterations / float(len(old_ps)))
    return ps


def stationary_distribution(Q, u, p_init, max_iter=50, convergence_criteria=1e-4):
    p = p_init
    for i in range(max_iter):
        new_p = np.dot(p, Q)
        new_p *= (1 - ALPHA)
        new_p[0, u] += ALPHA
        delta = 0 if convergence_criteria == 0 else np.sum(abs((new_p - p).data))
        p = new_p
        if delta < convergence_criteria:
            break
    return p, (i + 1)


def get_loss(ps, Ds, Ls, w):
    loss = 0
    for i, u in enumerate(ps):
        u_loss = 0
        u_updates = 0
        for d in Ds[u]:
            for l in Ls[u]:
                u_loss += h(ps[u][0, d] - ps[u][0, l])
                u_updates += 1
        loss += u_loss / u_updates

    loss /= len(ps)
    loss += REGULARIZATION_CONSTANT * np.sqrt(sum(wk ** 2 for wk in w.values()))

    return loss


def run(phi, w, Ds, Ls, old_ps):
    print "  w =", w
    print "  computing Q..."
    Q = get_Q(phi, w)
    print "  computing ps..."
    ps = get_ps(Q, old_ps)
    print "  computing loss..."
    loss = get_loss(ps, Ds, Ls, w)
    print "  loss =", loss
    return loss, ps


def train():
    phi = get_phi(True)

    print "Loading examples..."
    Ds, Ls = {}, {}
    examples = util.load_json('./data/train/examples.json')
    us = list(examples.keys())
    random.seed(0)
    random.shuffle(us)
    for u in us:
        D, L = set(), set()
        for b in examples[u]:
            (D if examples[u][b] == 1 else L).add(int(b))
        if len(D) > MAX_POSITIVE_EDGES_PER_USER:
            D = random.sample(D, MAX_POSITIVE_EDGES_PER_USER)
        if len(L) > MAX_NEGATIVE_EDGES_PER_USER:
            L = random.sample(L, MAX_POSITIVE_EDGES_PER_USER)
        if len(D) > 1 and len(L) > 10:
            Ds[int(u)] = list(D)
            Ls[int(u)] = list(L)
            if len(Ds) > NUM_TRAIN_USERS:
                break

    print "Setting initial conditions..."
    ps = {}
    for u in Ds:
        p = np.zeros(phi['bias'].shape[0])
        p[u] = 1.0
        ps[u] = sparse.csr_matrix(p)

    print "Training..."
    w = INITIAL_WEIGHTS
    best_loss = 100000
    for i in range(100):
        print "ITERATION " + str(i + 1) + ": base"
        base_loss, ps = run(phi, w, Ds, Ls, ps)
        if base_loss < best_loss:
            best_loss = base_loss
            util.write_json(w, './data/supervised_random_walks_weights.json')

        partials = {}
        for k in w:
            print "ITERATION " + str(i + 1) + ": " + k
            new_w = w.copy()
            new_w[k] += H
            new_loss, _ = run(phi, new_w, Ds, Ls, ps)
            partials[k] = (new_loss - base_loss) / H

            print partials[k] * LEARNING_RATE

        for (k, dwk) in partials.iteritems():
            w[k] -= LEARNING_RATE * dwk


def test():
    phi = get_phi(False)
    examples = util.load_json('./data/test/examples.json')
    w = util.load_json('./data/supervised_random_walks_weights.json')

    print "Computing Q and initializing..."
    Q = get_Q(phi, w)
    ps = {}
    for u in examples:
        p = np.zeros(phi['bias'].shape[0])
        p[int(u)] = 1.0
        ps[int(u)] = sparse.csr_matrix(p)
    get_ps(Q, ps, max_iter=20, convergence_criteria=0, log=True, examples=examples)

    print "Writing..."
    util.write_json(examples, './data/test/supervised_random_walks.json')


if __name__ == '__main__':
    train()
    test()
