
# coding: utf-8

# In[1]:

import numpy as np
import time


# In[19]:

#filename = "data/trainingSet.dat"
TRAIN = "../Data/trainingSet.dat"
TRAIN_LABELS = "../Data/trainingSetLabels.dat"

DEV = "../Data/developmentSet.dat"
DEV_LABELS = "../Data/developmentSetLabels.dat"

def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc(s=None):
    if 'startTime_for_tictoc' in globals():
        print s+":Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"

def get_articles(filename):
    with open(filename, "r") as inp:
        data = inp.read()

    data = data.split("\n")
    data.append("~~~~~ ")
    articles = []
    tmp = []
    for i in data:
        if i != "~~~~~ ":
            tmp.append(i)
        else:
            articles.append(tmp)
            tmp = []
    articles = articles[1:] 
    return articles

def get_articles_list(fname):
    with open(fname, 'r') as f:
        d = f.read()

    articles = d.split("\n~~~~~ \n") 
    articles_sents = [a.split("\n") for a in articles]
    return articles_sents

def load(fname):
    with open(fname, 'r') as f:
        d = f.read()
    return d

def write(fname, s):
    with open(fname, 'w') as f:
        f.write(s)

# In[21]:

#articles = get_articles(filename)
#print len(articles)
#print articles[0]


# In[27]:

def get_labels(filename):
    with open(filename, "r") as inp:
        labels = inp.read()
    labels = labels.split("\n")
    labels = filter(None, labels)
    labels = map(int, labels)
    return labels


# In[28]:

#label_fname = "data/trainingSetLabels.dat"
#labels = get_labels(label_fname)


# In[29]:

#print len(labels)
#print labels[-1]


# In[30]:

def split_articles(articles, labels):
    pos_article = []
    neg_article = []
    for i in range(len(labels)):
        if labels[i] == 1:
            pos_article.append(articles[i])
        else:
            neg_article.append(articles[i])
    return pos_article, neg_article


if __name__ == "__main__":
    tic()
    a1 = get_articles(TRAIN)
    toc("F1")
    tic()
    a = get_articles_list(TRAIN)
    toc("F2")


# In[31]:

#pos, neg = split_articles(articles, labels)


# In[33]:

#print pos[:5]


# In[34]:

#print neg[:5]


# In[ ]:



