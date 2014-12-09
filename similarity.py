import snap
import util
import networkx as nx
import datetime
from sets import Set
from scipy import spatial
import math
from collections import defaultdict


def main(example_file,graph_file,u_methods,u_outfiles,b_methods,b_outfiles):
	start = datetime.datetime.now()
	print "Loading examples..."
	examples = util.load_json(example_file)
	print "Loading graph..."
	G = snap.LoadEdgeList(snap.PUNGraph, graph_file, 0, 1)
	users(examples, G, u_methods, u_outfiles)
	business(examples, G, b_methods, b_outfiles)

def users(examples, G, methods, outfiles):
	hop2s = dict()
	nodes=[N.GetId() for N in snap.Nodes(G)]
	print "Getting 2 hops..."
	for u in examples:
		nodeid = int(u)
		if nodeid in nodes:
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
			if int(v) not in neighbors and int(v) in nodes:
				# Get neighbors of the business
				temp = snap.TIntV()
				snap.GetNodesAtHop(G, int(v), 1, temp, True)
				tempset = Set()
				for i in temp:
					tempset.add(i)
				neighbors[int(v)] = tempset
	print "Beginning computation..."

	for m,f in zip(methods,outfiles):
		u_sim=defaultdict(dict)
		for u in examples:
			for v in examples[u]:
				if (int(u) in nodes and int(v) in nodes):
					if (m=='common_neighbors'):
						u_sim[u][v] = common_neighbors(hop2s[int(u)], neighbors[int(v)])
					elif (m=='jaccard'):
						u_sim[u][v] = jaccard(hop2s[int(u)], neighbors[int(v)])
					elif (m=='adamic_adar'):
						u_sim[u][v] = adamic_adar(hop2s[int(u)], neighbors[int(v)],G)
				else:
					u_sim[u][v]=0
		util.write_json(u_sim, f)

def business(examples, G, methods, outfiles):
	hop2s = dict()
	nodes=[N.GetId() for N in snap.Nodes(G)]
	print "Getting 2 hops..."
	for u in examples:
		for v in examples[u]:
			if int(v) not in hop2s:
				nodeid = int(v)
		    	# Get nodes at hop 2
				if nodeid in nodes:
					hop2nodes = snap.TIntV()
					snap.GetNodesAtHop(G, nodeid, 2, hop2nodes, True)
					tempset = Set()
					for i in hop2nodes:
						tempset.add(i)
					hop2s[nodeid] = tempset
	print "Getting users..."
	neighbors = dict()
	for u in examples:
		if int(u) not in neighbors and int(u) in nodes:
			# Get neighbors of the user
			temp = snap.TIntV()
			snap.GetNodesAtHop(G, int(u), 1, temp, True)
			tempset = Set()
			for i in temp:
				tempset.add(i)
			neighbors[int(u)] = tempset
	print "Beginning computation..."
	for m,f in zip(methods,outfiles):
		b_sim=defaultdict(dict)
		for u in examples:
			for v in examples[u]:
				if (int(u) in nodes and int(v) in nodes):
					if (m=='common_neighbors'):
						b_sim[u][v] = common_neighbors(hop2s[int(v)], neighbors[int(u)])
					elif (m=='jaccard'):
						b_sim[u][v] = jaccard(hop2s[int(v)], neighbors[int(u)])
					elif (m=='adamic_adar'):
						b_sim[u][v] = adamic_adar(hop2s[int(v)], neighbors[int(u)],G)
				else:
					b_sim[u][v]=0
		util.write_json(b_sim, f)

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
		deg=G.GetNI(i).GetDeg()
		if (deg>1):
			sum += (math.log(deg))**-1
		else:
			sum+=0
	return sum

if __name__ == '__main__':
    main('./data/train/examples.json','./data/train/graph.txt',
             ['common_neighbors','jaccard','adamic_adar'],
             ['./data/train/u_cn.json', './data/train/u_jaccard.json',
					'./data/train/u_adamic.json'],
             ['common_neighbors','jaccard','adamic_adar'],
		  ['./data/train/b_cn.json', './data/train/b_jaccard.json',
					'./data/train/b_adamic.json'],)
    main('./data/test/examples.json','./data/test/graph.txt',
             ['common_neighbors','jaccard','adamic_adar'],
             ['./data/test/u_cn.json', './data/test/u_jaccard.json',
					'./data/test/u_adamic.json'],
             ['common_neighbors','jaccard','adamic_adar'],
		  ['./data/test/b_cn.json', './data/test/b_jaccard.json',
					'./data/test/b_adamic.json'],)
