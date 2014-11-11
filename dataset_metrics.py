import snap
import util
import os
import networkx as nx
from networkx.algorithms import bipartite

class KeyToInt():
    def __init__(self):
        self._n = 0
        self._map = {}

    def __getitem__(self, key):
        if key not in self._map:
            self._n += 1
            self._map[key] = self._n
        return self._map[key]

def reviews_iterator(path='./data/provided/yelp_academic_dataset_review.json'):
    return util.logged_loop(util.load_json_lines(path),
                            util.LoopLogger(100000, util.lines_in_file(path), True))
def get_user_nodes():
	print "blah"
def get_review_nodes():
	print "blah"
def get_business_nodes():
	print "blah"
def get_degree_distribution(graph):
	dist = dict()
	for node in graph.Nodes():
		dist[node.GetId()] = node.GetOutDeg()
	degdist = dict()
	sum = 0
	for n in dist:
		sum += 1
		if dist[n] not in degdist:
			degdist[dist[n]] = 1
		else:
			degdist[dist[n]] += 1
	# Normalize
	with open('degreedist.txt', 'w') as f:
		for n in degdist:
			normalized = float(degdist[n])/float(sum)
			f.write(str(n) + "\t" + str(normalized) + '\n')
		f.close()

def get_clustering_coefficient(graph):
	print "Clustering coefficient of graph 1: %.9f"  % snap.GetClustCf(graph, -1)
def get_alpha(file):
	print "TODO"
def get_diameter(graph):
	diam = snap.GetBfsFullDiam(graph, 100, False)
	print "Diameter is: %f" % diam

def make_complete_dataset(out_dir):
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	id_to_nid = KeyToInt()

	print "Building complete set of nodes..."
	nids = set()
	for review in reviews_iterator():
		nids.add(id_to_nid[review['user_id']])
		nids.add(id_to_nid[review['business_id']])

	print "Building graph..."
	with open(out_dir + 'completegraph.txt', 'w') as graph:
		for review in reviews_iterator():
			user_key = id_to_nid[review['user_id']]
			business_key = id_to_nid[review['business_id']]
			if user_key in nids and business_key in nids:
				graph.write("{:} {:}\n".format(user_key, business_key))

def get_metrics(out_dir):
	""" Create three graphs representing the 3 networks """
	#G=nx.DiGraph()
	#G=nx.read_edgelist(out_dir + 'metricscompletegraph.txt')
	#print(nx.clustering(G,0))
	print "Loading edges..."
	G = snap.LoadEdgeList(snap.PNGraph,out_dir + 'metricscompletegraph.txt',0,1,' ')
	#get_clustering_coefficient(G)
	#print snap.IsConnected(G)
	#get_degree_distribution(G)
	get_diameter(G)
	#get_alpha('degreedist.txt')

if __name__ == '__main__':
	#make_complete_dataset('./data/metrics')
	get_metrics('./data/metrics/')