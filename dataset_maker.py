import datetime
import os
import random
import snap
import util
from collections import defaultdict, Counter


class KeyToInt():
    def __init__(self):
        self._n = -1
        self._map = {}

    def __getitem__(self, key):
        if key not in self._map:
            self._n += 1
            self._map[key] = self._n
        return self._map[key]


def get_date(review):
    return datetime.date(*map(int, review['date'].split('-')))


def reviews_iterator(path='./data/provided/yelp_academic_dataset_review.json'):
    return util.logged_loop(util.load_json_lines(path),
                            util.LoopLogger(100000, util.lines_in_file(path), True))


def write_node_data(nid_f, nids, infile, outfile):
    return util.write_json({nid_f(datum): datum for datum in util.load_json_lines(infile)
                            if nid_f(datum) in nids}, outfile)


def print_dataset_stats(data_dir):
    G = snap.LoadEdgeList(snap.PUNGraph, data_dir + 'graph.txt', 0, 1)
    print "Num nodes:", G.GetNodes()
    print "Num edges:", G.GetEdges()

    n_users = len(util.load_json(data_dir + "user.json"))
    n_businesses = len(util.load_json(data_dir + "business.json"))
    n_edges = util.lines_in_file(data_dir + "new_edges.txt")
    print "({:} users) * ({:} businesses) = {:.3e} candidate edges".format(
        n_users, n_businesses, n_users * n_businesses)
    print "{:} edges, {:0.5f}% of candidate edges".format(n_edges, 100 * n_edges /
                                                          float(n_users * n_businesses))


def make_examples_simple(data_dir, n_users, negative_examples_per_user=10):
    G = snap.LoadEdgeList(snap.PUNGraph, data_dir + 'graph.txt', 0, 1)
    new_edges = defaultdict(dict)
    with open(data_dir + 'new_edges.txt') as f:
        for line in f:
            u, b = map(int, line.split())
            new_edges[u][b] = 1

    businesses = map(int, util.load_json(data_dir + 'business.json').keys())
    examples = defaultdict(dict)
    users = random.sample([NI.GetId() for NI in G.Nodes()], n_users)
    for u in users:
        examples[u] = new_edges[u]
        for i in range(negative_examples_per_user):
            b = random.choice(businesses)
            examples[u][b] = 0

    p, n = 0, 0
    for u in examples:
        for b in examples[u]:
            p += examples[u][b]
            n += 1 - examples[u][b]
    print "Positive:", p
    print "Negative:", n
    print "Data skew:", p / float(p + n)
    print "Sampling rate:", negative_examples_per_user / float(len(businesses))

    print "Writing examples..."
    util.write_json(examples, data_dir + 'examples_simple.json')


def make_examples(data_dir, n_users=5000, min_degree=1, negative_sample_rate=0.01,
                  min_active_time=None, new_edge_only=False):
    print "Loading data..."
    # TODO: switch to networkx?
    G = snap.LoadEdgeList(snap.PUNGraph, data_dir + 'graph.txt', 0, 1)
    with open(data_dir + 'new_edges.txt') as f:
        edges = {tuple(map(int, line.split())) for line in f}
    new_edge_count = Counter()
    for (u, b) in edges:
        new_edge_count[u] += 1
    review_data = util.load_json(data_dir + 'review.json')
    n_businesses = len(util.load_json(data_dir + "business.json"))

    recently_active_users = []
    other_users = []
    print "Getting candidate set of users..."
    users = []
    for Node in util.logged_loop(G.Nodes(), util.LoopLogger(50000, G.GetNodes(), True)):
        u = Node.GetId()
        if new_edge_only and not u in new_edge_count:
            continue
        if str(u) not in review_data or Node.GetOutDeg() < min_degree:
            continue
        if min_active_time:
            recent_review = False
            for b in review_data[str(u)]:
                if (int(u), int(b)) in edges:
                    continue
                for r in review_data[str(u)][b]:
                    if get_date(r) > min_active_time:
                        users.append(u)
                        recently_active_users.append(u)
                        recent_review = True
                        break
                if recent_review:
                    break
            if not recent_review:
                other_users.append(u)
        else:
            users.append(u)

    if min_active_time:
        recent_positive = sum(new_edge_count[u] for u in recently_active_users)
        recent_examples = len(recently_active_users) * n_businesses
        other_positive = sum(new_edge_count[u] for u in other_users)
        other_examples = len(other_users) * n_businesses
        print "Positives retained from recently active filter:", \
            recent_positive / float(recent_positive + other_positive)
        print "Negatives retained from recently active filter:", \
            (recent_examples - recent_positive) / \
            float(recent_examples - recent_positive + other_examples - other_positive)

    random.seed(0)
    users = random.sample(users, n_users)

    print "Getting candidate set of edges..."
    examples = defaultdict(dict)
    for u in util.logged_loop(users, util.LoopLogger(50, n_users, True)):
        candidate_businesses = snap.TIntV()
        snap.GetNodesAtHop(G, u, 3, candidate_businesses, True)
        for b in candidate_businesses:
            if (u, b) in edges:
                examples[u][b] = 1
            elif random.random() < negative_sample_rate:
                examples[u][b] = 0

    hop3_positives = 0
    for u in examples:
        for b in examples[u]:
            hop3_positives += examples[u][b]
    hop3_examples = sum(len(examples[u]) for u in examples)
    n_positives = sum([new_edge_count[u] for u in users])
    n_examples = len(users) * n_businesses
    print "Positives retained from hop3 filter:", hop3_positives / float(n_positives)
    print "Negatives retained from hop3 filter:", (hop3_examples - hop3_positives) / \
            (negative_sample_rate * float(n_examples - n_positives))
    print "Data skew:", hop3_positives / float(hop3_examples)

    print "Writing examples..."
    util.write_json(examples, data_dir + 'examples.json')


def make_dataset(t1, t2, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # we need to map the ids in the yelp data to ints since snap only allows ints as node ids
    id_to_nid = KeyToInt()

    print "Building set of nodes..."
    nids = set()
    for review in reviews_iterator():
        if get_date(review) < t1:
            nids.add(id_to_nid['u' + review['user_id']])
            nids.add(id_to_nid['b' + review['business_id']])

    print "Building user data..."
    write_node_data(lambda user_data: id_to_nid['u' + user_data['user_id']], nids,
                    './data/provided/yelp_academic_dataset_user.json',
                    out_dir + 'user.json')

    print "Building business data..."
    write_node_data(lambda business_data: id_to_nid['b' + business_data['business_id']], nids,
                    './data/provided/yelp_academic_dataset_business.json',
                    out_dir + 'business.json')

    print "Building graph..."
    with open(out_dir + 'graph.txt', 'w') as graph, \
            open(out_dir + 'new_edges.txt', 'w') as new_edges:
        review_data = defaultdict(lambda: defaultdict(list))
        for review in reviews_iterator():
            user_key = id_to_nid['u' + review['user_id']]
            business_key = id_to_nid['b' + review['business_id']]
            if user_key in nids and business_key in nids:
                date = get_date(review)
                if date < t1:
                    review_data[user_key][business_key].append(review)
                    graph.write("{:} {:}\n".format(user_key, business_key))
                elif date < t2:
                    new_edges.write("{:} {:}\n".format(user_key, business_key))

        for u in review_data:
            for b in review_data[u]:
                review_data[u][b] = sorted(review_data[u][b], key=get_date, reverse=True)

        util.write_json(review_data, out_dir + "review.json")


if __name__ == '__main__':
    make_dataset(datetime.date(2012, 1, 1), datetime.date(2012, 7, 1), './data/train/')
    make_examples('./data/train/', n_users=10000, min_degree=1, negative_sample_rate=0.01,
                  min_active_time=datetime.date(2011, 7, 1), new_edge_only=False)

    make_dataset(datetime.date(2013, 1, 1), datetime.date(2013, 7, 1), './data/test/')
    make_examples('./data/test/', n_users=10000, min_degree=1, negative_sample_rate=0.01,
                  min_active_time=datetime.date(2012, 7, 1), new_edge_only=False)