# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 00:34:27 2014

@author: kam
"""
import util
from sklearn import svm
import networkx as nx
import time

def svm_classifier():
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
    
    svm_class=svm.SVC(kernel='linear')
    svm_class.fit(train_features,train_target)
    print "finished model fitting..."
    train=util.load_json("./data/train/supervised_examples2.json")
    test_features=[]
    test_target=[]
    for u in train:
        u=int(u)
        for b,e in train[str(u)].iteritems():
            b=int(b)
            e=int(e)
            if ({u,b}.issubset(nodes)):
                test_features.append([degrees[u],degrees[b]])
                test_target.append(e)
    
    result=svm_class.predict(test_features)
    confusionMatrix(test_target,result)
    
    print "Finished processing linear kernel"
    poly_svm_class=svm.SVC(kernel='poly')
    poly_svm_class.fit(train_features,train_target)
    print "finished poly model fitting..."
    poly_result=poly_svm_class.predict(test_features)
    print "finished poly model prediction..."
    confusionMatrix(test_target,poly_result)

def confusionMatrix(tgt,res):
    tp=tn=fp=fn=0
    for item in zip(tgt,res):
        if (item[0]==1 and item[1]==1):
            tp+=1
        elif(item[0]==0 and item[1]==0):
            tn+=1
        elif(item[1]==0 and item[1]==1):
            fp+=1
        else:
            fn+=1
    print "True Positive:{0}; True Negative={1}; False Positive={2}; \
        False Negative={3}".format(tp,tn,fp,fn)

if __name__=="__main__":
    now=time.time()
    print time.localtime()
    svm_classifier()
    print "The process has taken {:} secs".format(time.time()-now)
    
<<<<<<< Updated upstream
    
=======
    
>>>>>>> Stashed changes
