import datetime
import os
import random
import snap
import util
from collections import defaultdict


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


def make_examples(data_dir, n_users=5000, min_degree=1, negative_sample_rate=0.01,
                  min_active_time=None):
    """Creates a set of edges to be used as examples from a dataset. Using all (user, business)
    pairs as candidate edges is computationally infeasible, so we heuristically pick a set of edges
    that are likely to exist in the future.

    This is done by picking n_users users that have degree at least min_degree and have written a
    review after min_active_time. These users are linked to all businesses that are of distance 3
    from them.
    """
    print "Loading data..."
    # TODO: switch to networkx?
    G = snap.LoadEdgeList(snap.PUNGraph, data_dir + 'graph.txt', 0, 1)
    with open(data_dir + 'new_edges.txt') as f:
        edges = {tuple(map(int, line.split())) for line in f}
    review_data = util.load_json(data_dir + 'review.json')

    print "Getting candidate set of users..."
    users = []
    for Node in util.logged_loop(G.Nodes(), util.LoopLogger(50000, G.GetNodes(), True)):
        u = Node.GetId()
        if str(u) not in review_data or Node.GetOutDeg() < min_degree:
            continue
        if min_active_time:
            recent_review = False
            for b in review_data[str(u)]:
                for r in review_data[str(u)][b]:
                    if get_date(r) > min_active_time:
                        users.append(u)
                        recent_review = True
                        break
                if recent_review:
                    break
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

    print "Writing examples..."
    util.write_json(examples, data_dir + 'examples.json')


def make_dataset(t1, t2, out_dir):
    """Creates a dataset in out_dir consisting of
        - A snapshot of the review graph at time t1
        - All new edges that are added between times t1 and t2
        - Data for the relevant users, businesses, and reviews
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # we need to map the ids in the yelp data to ints since snap only allows ints as node ids
    id_to_nid = KeyToInt()

    print "Building set of nodes..."
    nids = set()
    for review in reviews_iterator():
        if get_date(review) < t1:
            nids.add(id_to_nid[review['user_id']])
            nids.add(id_to_nid[review['business_id']])

    print "Building user data..."
    write_node_data(lambda user_data: id_to_nid[user_data['user_id']], nids,
                    './data/provided/yelp_academic_dataset_user.json',
                    out_dir + 'user.json')

    print "Building business data..."
    write_node_data(lambda business_data: id_to_nid[business_data['business_id']], nids,
                    './data/provided/yelp_academic_dataset_business.json',
                    out_dir + 'business.json')

    print "Building graph..."
    with open(out_dir + 'graph.txt', 'w') as graph, \
            open(out_dir + 'new_edges.txt', 'w') as new_edges:
        review_data = defaultdict(lambda: defaultdict(list))
        for review in reviews_iterator():
            user_key = id_to_nid[review['user_id']]
            business_key = id_to_nid[review['business_id']]
            if user_key in nids and business_key in nids:
                review_data[user_key][business_key].append(review)
                date = get_date(review)
                if date < t1:
                    graph.write("{:} {:}\n".format(user_key, business_key))
                elif date < t2:
                    new_edges.write("{:} {:}\n".format(user_key, business_key))
        util.write_json(review_data, out_dir + "review.json")


if __name__ == '__main__':
    make_dataset(datetime.date(2013, 1, 1), datetime.date(2013, 7, 1), './data/train/')
    make_examples('./data/train/', n_users=5000, min_degree=5,
                  min_active_time=datetime.date(2012, 7, 1))
