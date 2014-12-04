# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 15:16:34 2014

@author: kam
"""
import util
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from collections import Counter
from collections import defaultdict
import datetime

def supervised_methods(methods):
    num_features=8
    #t1 dates for make dataset: 2012-01-01 to 2012-07-01; 
    #for examples it is 2011-07-01
    train=build_features("./data/train/t1_user.json", 
                         "./data/train/t1_business.json",
                         "./data/train/t1_supervised_examples.json",
                         "./data/train/t1_graph.txt",num_features,
                         datetime.date(2011,7,1))
    #t2 dates fot the make dataset: 2013-01-01 to 2013-07-01; 
                         #for examples it is 2012-07-01
    test=build_features("./data/train/t2_user.json", 
                        "./data/train/t2_business.json",
                         "./data/train/t2_supervised_examples.json",
                         "./data/train/t2_graph.txt",num_features,
                         datetime.date(2013,7,1),True)
    for method in methods:
        clf=None
        if method=="RandomForest":
            clf=RandomForestClassifier(n_estimators=100,
                        max_features=num_features, oob_score=True)
        elif method=="GBM":
            clf=GradientBoostingClassifier(n_estimators=100,
                                           max_features=num_features)
        else:
            continue
        
        clf=clf.fit(train["features"],train["target"])
        probs=clf.predict_proba(test["features"])
        prob_json=test["probs"]
        for u in prob_json:
            for b in prob_json[u]:
                prob_json[u][b]=float(probs[prob_json[u][b]][1])
        util.write_json(prob_json,"./data/train/"+method+".json")
        with open("./data/train/"+method+"_scores.txt","w") as f:
            f.write("===feat. importance==="+str(clf.feature_importances_)+"\n")
            f.flush()
        f.close()

def build_features(user_json_file, biz_json_file, data_json_file, 
                   graph_edges_file,n_feat,st_date,prob_flag=False):
    users=util.load_json(user_json_file)
    biz=util.load_json(biz_json_file)
    data=util.load_json(data_json_file)
    G=nx.read_edgelist(graph_edges_file,nodetype=int)
    degrees=G.degree()
    nodes=set(G.nodes())
    features=[]
    target=[]
    user_dict={}
    for u,v in users.iteritems():
        user_dict[v["user_id"]]=u
    #features in the order - user degree, biz degree,
    #average biz stars, average biz review count, user review count, 
    #number of friends, user average stars, total votes, friend reviewed
    num_features=n_feat # TODO: I have to write the friend reviewed function 
    prob_list=defaultdict(dict)
    for u in data:
        u=int(u)
        for b,e in data[str(u)].iteritems():
            b=int(b)
            e=int(e)
            feat=np.zeros(num_features,dtype=float)
            if ({u,b}.issubset(nodes)):
                feat[0]=degrees[u]
                feat[1]=degrees[b]
                if str(b) in biz:
                    feat[2]=biz[str(b)]["stars"]
                    feat[3]=biz[str(b)]["review_count"]
                else:
                    continue
                if str(u) in users:
                    feat[4]=users[str(u)]["review_count"]
                    feat[5]=len(users[str(u)]["friends"])
                    feat[6]=users[str(u)]["average_stars"]
                    votes=Counter(users[str(u)]["votes"])
                    feat[7]=votes["funny"]+votes["useful"]+votes["cool"]
                else:
                    continue
                #TODO: this is not optimized yet and I am working on it to better this
                #feat[8]=numFriendsReviewed(G,user_dict,b, 
                    #users[str(u)]["friends"],data,st_date)
                features.append(feat)
                target.append(e)
                if (prob_flag):
                    prob_list[str(u)][str(b)]=len(target)-1
    
    res={"features":features, "target":target}
    if (prob_flag):
        res["probs"]=prob_list
    return res

def numFriendsReviewed(graph,user_dict,biz_node,
                       friends_list,rev_dict,st_date):
    if len(friends_list)==0:
        return 0
    else:
        tot=0
        for friend in friends_list:
            if friend in user_dict:
                friend_node=user_dict[friend]
                if (int(user_dict[friend]),biz_node) in graph.edges():
                    if str(friend_node) in rev_dict:
                        if str(biz_node) in rev_dict[str(friend_node)]:
                            rev_date=datetime.date(*map(int, 
                                rev_dict[str(friend_node)][str(biz_node)]['date'].split('-')))
                            if (rev_date<st_date):
                                tot+=1
        return tot
if __name__=='__main__':
    supervised_methods(["RandomForest","GBM"])
    
