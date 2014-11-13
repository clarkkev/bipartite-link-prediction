# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 20:10:40 2014

@author: kam
"""
from collections import defaultdict
from sklearn.linear_model import SGDClassifier
import networkx as nx
import util

def sgd():
    G=nx.read_edgelist("./data/train/graph1.txt",nodetype=int)
    degrees=G.degree()
    nodes=set(G.nodes())
    
    train=util.load_json("./data/train/supervised_examples1.json")
    train_features=[]
    train_target=[]
    
    for u in train:
        u=int(u)
        for b,e in train[str(u)].iteritems():
            b=int(b)
            e=int(e)
            if ({u,b}.issubset(nodes)):
                train_features.append([degrees[u],degrees[b]])
                train_target.append(e)
    sgd_class=SGDClassifier(loss="log",n_iter=100)
    sgd_class.fit(train_features,train_target)

    test=util.load_json("./data/train/supervised_examples2.json")
    test_features=[]
    test_target=[]
    res=defaultdict(dict)
    for u in test:
        u=int(u)
        for b,e in test[str(u)].iteritems():
            b=int(b)
            e=int(e)
            if ({u,b}.issubset(nodes)):
                test_features.append([degrees[u],degrees[b]])
                test_target.append(e)
                res[str(u)][str(b)]=len(test_target)-1
    res_probs=sgd_class.predict_proba(test_features)
    for u in res:
        for b in res[u]:
            res[u][b]=res_probs[res[u][b]][1]
    util.write_json(res,"./data/train/sgd_100.json")

if __name__=="__main__":
    sgd()
