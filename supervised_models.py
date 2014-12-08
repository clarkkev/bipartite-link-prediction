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
import time
import random

random.seed(1000)
def supervised_methods(methods):
    num_features=14
    #train dates for make dataset: 2012-01-01 to 2012-07-01;
    #for examples it is 2011-07-01
    train=build_features("./data/train/user.json",
                         "./data/train/business.json",
                         "./data/train/examples.json",
                         "./data/train/graph.txt",
				   "./data/train/user_adamic_adar.json",
				   "./data/train/biz_adamic_adar.json",
				   "./data/train/user_cn.json",
				   "./data/train/biz_cn.json",
				   "./data/train/user_jaccard.json",
				   "./data/train/biz_jaccard.json",
				   num_features,datetime.date(2011,7,1))
    #test dates fot the make dataset: 2013-01-01 to 2013-07-01;
                         #for examples it is 2012-07-01
    test=build_features( "./data/test/user.json",
                         "./data/test/business.json",
                         "./data/test/examples.json",
                         "./data/test/graph.txt",
				   "./data/test/user_adamic_adar.json",
				   "./data/test/biz_adamic_adar.json",
				   "./data/test/user_cn.json",
				   "./data/test/biz_cn.json",
				   "./data/test/user_jaccard.json",
				   "./data/test/biz_jaccard.json",
				   num_features,datetime.date(2012,7,1),True)

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

        util.write_json(prob_json,"./data/results/"+method+".json")
        with open("./data/results/"+method+"_scores.txt","w") as f:
            f.write("===feat. importance==="+str(clf.feature_importances_)+"\n")
            f.flush()
        f.close()

def build_features(user_json_file, biz_json_file,
		data_json_file, graph_edges_file, u_adamic_score_file,
		b_adamic_score_file, u_cn_score_file, b_cn_score_file,
		u_jaccard_score_file, b_jaccard_score_file, n_feat, st_date, prob_flag=False):
    #rev=util.load_json(rev_json_file)
    users=util.load_json(user_json_file)
    biz=util.load_json(biz_json_file)
    data=util.load_json(data_json_file)
    G=nx.read_edgelist(graph_edges_file,nodetype=int)
    u_adamic_score=util.load_json(u_adamic_score_file)
    b_adamic_score=util.load_json(b_adamic_score_file)
    u_cn_score=util.load_json(u_cn_score_file)
    b_cn_score=util.load_json(b_cn_score_file)
    u_jaccard_score=util.load_json(u_jaccard_score_file)
    b_jaccard_score=util.load_json(b_jaccard_score_file)
    degrees=G.degree()
    nodes=set(G.nodes())
    features=[]
    target=[]
    """
    user_dict={}
    for u,v in users.iteritems():
        user_dict[v["user_id"]]=u
    """
    #features in the order - user degree, biz degree,
    #average biz stars, average biz review count, user review count,
    #number of friends, user average stars, total votes, friend reviewed

    prob_list=defaultdict(dict)
    for u in data:
        u=int(u)
        for b,e in data[str(u)].iteritems():
            b=int(b)
            e=int(e)
            feat=np.zeros(n_feat,dtype=float)
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
			feat[8]=u_adamic_score[str(u)][str(b)]
			feat[9]=b_adamic_score[str(u)][str(b)]
			feat[10]=u_cn_score[str(u)][str(b)]
			feat[11]=b_cn_score[str(u)][str(b)]
			feat[12]=u_jaccard_score[str(u)][str(b)]
			feat[13]=b_jaccard_score[str(u)][str(b)]
			"""
			#This feature is discarded because it has shown no impact
			feat[8]=numFriendsRev(users,rev,user_dict,u,b,st_date)
			"""
			features.append(feat)
			target.append(e)
			if (prob_flag):
				prob_list[str(u)][str(b)]=len(target)-1
    res={"features":features, "target":target}
    if (prob_flag):
        res["probs"]=prob_list
    return res


"""Testing has been done on applying this feature but because of only handful of
cases where the friends also reviewed the business, the sparsity of this feature
has not shown any impact; hence the usage of this feature is discarded"""
def numFriendsRev(users,rev,udict,u,b,dt):
    """dt is the starting date of the examples, here - 2012-07-01"""
    """users, rev are the user.json and review.json files"""
    """udict is the user dict; u is the current user, b is the current biz"""
    tot=0
    if u in users:
        friends=[udict[f] for f in users[u]['friends'] if f in udict]
        for friend in friends:
            if friend in rev:
                if b in rev[friend]:
                    for i in xrange(0,len(rev[friend][b])):
                        if (isPriorDate(rev[friend][b][i]['date'],dt)):
                            tot+=1
    return tot

def isPriorDate(str_date,dt):
    date_vals=map(int,str_date.split("-"))
    dt_params=[dt.year,dt.month,dt.day]
    str_tot=0
    dt_tot=0
    for x in range(0,len(date_vals)):
        mul=10**(4/(2**x))
        str_tot+=mul*date_vals[x]
        dt_tot+=mul*dt_params[x]
    return (str_tot<dt_tot)

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
    start=time.clock()
    supervised_methods(["RandomForest","GBM"])

    #test(datetime.date(2012,7,1))
    print str(time.clock()-start) + "secs"

