# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 20:10:40 2014

@author: kam
"""
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
import networkx as nx
import util

def bayes():
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
    gnb_class=GaussianNB()
    gnb_class.fit(train_features,train_target)

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
    res_probs=gnb_class.predict(test_features)
    for u in res:
        for b in res[u]:
            res[u][b]=res_probs[res[u][b]]
    util.write_json(res,"./data/train/gnb.json")
    print "Total Correct Values %d" %map(lambda x: x[0]==x[1], zip(test_target,res_probs)).count(True)
    print "Total Incorrect Values %d" %map(lambda x: x[0]==x[1], zip(test_target,res_probs)).count(False)
    print "Total Correct ones %d" %len([1 for x in zip(test_target,res_probs) if (x[0]==1 and x[0]==x[1])])
    print "Total Incorrect ones %d" %len([1 for x in zip(test_target,res_probs) if (x[1]==1 and x[0]!=x[1])])
    print "Total Values %d" %len(res_probs)

if __name__=="__main__":
    bayes()