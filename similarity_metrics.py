import snap
import util
import networkx as nx
import datetime
from sets import Set
from scipy import spatial
import math

def main():
	start = datetime.datetime.now()
	print "Loading examples..."
	examples = util.load_json('./data/test/examples.json')
	print "Loading graph..."
	G = snap.LoadEdgeList(snap.PUNGraph, './data/test/graph.txt', 0, 1)
	#users(examples, G)
	business(examples, G)

def users(examples, G):
	hop2s = dict()
	print "Getting 2 hops..."
	for u in examples:
		nodeid = int(u)
    	# Get nodes at hop 2
		hop2nodes = snap.TIntV()
		snap.GetNodesAtHop(G, nodeid, 2, hop2nodes, True)
		tempset = Set()
		for i in hop2nodes:
			tempset.add(i)
		hop2s[nodeid] = tempset
	print "Getting businesses..."
	neighbors = dict()
	for u in examples:
		for v in examples[u]:
			if int(v) not in neighbors:
				# Get neighbors of the business
				temp = snap.TIntV()
				snap.GetNodesAtHop(G, int(v), 1, temp, True)
				tempset = Set()
				for i in temp:
					tempset.add(i)
				neighbors[int(v)] = tempset
	print "Beginning computation..."
	for u in examples:
		for v in examples[u]:
			examples[u][v] = adamic_adar(hop2s[int(u)], neighbors[int(v)], G)
	util.write_json(examples, './data/test/common_neighbors_users.json')

def business(examples, G):
	hop2s = dict()
	print "Getting 2 hops..."
	for u in examples:
		for v in examples[u]:
			if int(v) not in hop2s:
				nodeid = int(v)
		    	# Get nodes at hop 2
				hop2nodes = snap.TIntV()
				snap.GetNodesAtHop(G, nodeid, 2, hop2nodes, True)
				tempset = Set()
				for i in hop2nodes:
					tempset.add(i)
				hop2s[nodeid] = tempset
	print "Getting users..."
	neighbors = dict()
	for u in examples:
		if int(u) not in neighbors:
			# Get neighbors of the user
			temp = snap.TIntV()
			snap.GetNodesAtHop(G, int(u), 1, temp, True)
			tempset = Set()
			for i in temp:
				tempset.add(i)
			neighbors[int(u)] = tempset
	print "Beginning computation..."
	for u in examples:
		for v in examples[u]:
			examples[u][v] = adamic_adar(hop2s[int(v)], neighbors[int(u)], G)
	util.write_json(examples, './data/test/adamic_adar_business.json')

def jaccard(setone, settwo):
	intersection = len(setone.intersection(settwo))
	union = len(setone.union(settwo))
	return float(intersection)/float(union)

def common_neighbors(setone, settwo):
    return len(setone.intersection(settwo))

def adamic_adar(setone, settwo, G):
	intersection = setone.intersection(settwo)
	sum = 0
	for i in intersection:
		# Get neighbors count
		sum += (math.log(G.GetNI(i).GetDeg()))**-1
	return sum

if __name__ == '__main__':
    main()
