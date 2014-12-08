# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 20:10:40 2014
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
@author: kam
"""
from collections import defaultdict
from sklearn.linear_model import SGDClassifier
import networkx as nx
import util
<<<<<<< Updated upstream

def sgd():
    G=nx.read_edgelist("./data/train/graph1.txt",nodetype=int)
    degrees=G.degree()
    nodes=set(G.nodes())
    
    train=util.load_json("./data/train/supervised_examples1.json")
=======
import random

def sgd():
    G=nx.read_edgelist("./data/train/graph.txt",nodetype=int)
    degrees=G.degree()
    nodes=set(G.nodes())
    
    train=util.load_json("./data/train/supervised_examples.json")
>>>>>>> Stashed changes
    train_features=[]
    train_target=[]
    
    for u in train:
        u=int(u)
        for b,e in train[str(u)].iteritems():
            b=int(b)
            e=int(e)
            if ({u,b}.issubset(nodes)):
<<<<<<< Updated upstream
                train_features.append([degrees[u],degrees[b]])
=======
                train_features.append([degrees[u],degrees[b], e])
>>>>>>> Stashed changes
                train_target.append(e)
    sgd_class=SGDClassifier(loss="log",n_iter=100)
    sgd_class.fit(train_features,train_target)

<<<<<<< Updated upstream
    test=util.load_json("./data/train/supervised_examples2.json")
=======
    test=util.load_json("./data/train/supervised_examples1.json")
>>>>>>> Stashed changes
    test_features=[]
    test_target=[]
    res=defaultdict(dict)
    for u in test:
        u=int(u)
        for b,e in test[str(u)].iteritems():
            b=int(b)
            e=int(e)
            if ({u,b}.issubset(nodes)):
<<<<<<< Updated upstream
                test_features.append([degrees[u],degrees[b]])
                test_target.append(e)
                res[str(u)][str(b)]=len(test_target)-1
    res_probs=sgd_class.predict_proba(test_features)
=======
                test_features.append([degrees[u],degrees[b], e])
                test_target.append(e)
                res[str(u)][str(b)]=len(test_target)-1
    res_probs=sgd_class.predict_proba(test_features)
    print sgd_class.coef_, sgd_class.intercept_
>>>>>>> Stashed changes
    for u in res:
        for b in res[u]:
            res[u][b]=res_probs[res[u][b]][1]
    util.write_json(res,"./data/train/sgd_100.json")

if __name__=="__main__":
<<<<<<< Updated upstream
    sgd()
=======
    sgd()
>>>>>>> Stashed changes
