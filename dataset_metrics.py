import math
import os
import snap
import util

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

def get_nodes(graph):
	print "Graphp has %d nodes" % graph.GetNodes()

def get_edges(graph):
	print "Graph has %d edges" % graph.GetEdges()

def get_degree_distribution(graph):
	print "Generating txt file with graph's degree distribution"
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

def get_alpha(graph):
	freq = dict()
	for node in graph.Nodes():
		degreeCount = node.GetOutDeg()
		if degreeCount not in freq:
			freq[degreeCount] = 1
		else:
			freq[degreeCount] += 1
	sum = 0
	n = 0
	xmin = 4
	for x in freq:
		sum += math.log(float(float(freq[x])/float(xmin)))
		n += 1
	alpha = 1 + n*(sum**-1)
	print "MLE Alpha: %.9f" % alpha

def get_average_degree(graph):
	edgeCount = 0
	nodeCount = 0
	for node in graph.Nodes():
		edgeCount += node.GetOutDeg()
		nodeCount += 1
	print "Average degree of a node: %f" % (float(edgeCount)/float(nodeCount))

def get_diameter(graph):
	diam = snap.GetBfsFullDiam(graph, 100, False)
	print "Diameter is: %f" % diam

def get_connected_info(graph):
	cd = snap.TIntPrV()
	snap.GetWccSzCnt(graph, cd)
	for comp in cd:
		print "Size: %d - Number of Components: %d" % (comp.GetVal1(), comp.GetVal2())

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
	with open(out_dir + 'metricscompletegraph.txt', 'w') as graph:
		for review in reviews_iterator():
			user_key = id_to_nid[review['user_id']]
			business_key = id_to_nid[review['business_id']]
			if user_key in nids and business_key in nids:
				graph.write("{:} {:}\n".format(user_key, business_key))

def get_metrics(out_dir):
	print "Loading edges..."
	G = snap.LoadEdgeList(snap.PNGraph,out_dir + 'metricscompletegraph.txt',0,1,' ')
	get_nodes(G)
	get_edges(G)
	print "Graph is connected? %s" % snap.IsConnected(G)
	get_degree_distribution(G)
	get_diameter(G)
	get_alpha(G)
	get_average_degree(G)
	get_connected_info(G)

if __name__ == '__main__':
	make_complete_dataset('./data/metrics/')
	get_metrics('./data/metrics/')