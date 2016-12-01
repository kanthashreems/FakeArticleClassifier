import numpy as np
from utils import *
from collections import Counter
import matplotlib.pyplot as plt
import pdb
import math

# def get_entropy(count_list):
#     count_arr = np.array(count_list)
#     s = sum(count_arr)
#     if s==0:
#         return 1
#     prob_arr = (1.0/s)*count_arr
#     return -sum(np.multiply(prob_arr, np.log2(prob_arr)))

def get_entropy(prob_arr):
	return -sum(np.multiply(prob_arr, np.log2(prob_arr)))

def get_perplexity_of_article(article):
	article_str = " ".join(article)
	article_ngram_list = article_str.split()
	perp = get_perplexity_of_article_ngram_list(article_ngram_list)
	return perp

def get_perplexity_of_article_ngram_list(article_ngram_list):
	c = Counter(article_ngram_list)
	counts = np.array(c.values())
	probs = counts/float(np.sum(counts))
	entropy = get_entropy(probs)
	perp = math.pow(2,entropy)
	# perp = entropy
	return perp

def get_perplexity_of_article_ngram(article, n):
	article = [s.replace("<s>","").replace("</s>","") for s in article]
	# print article
	ngrams_list = get_ngrams_for_article(" <stop> ".join(article), n)
	# print ngrams_list
	perp = get_perplexity_of_article_ngram_list(ngrams_list)
	return perp

def get_perplexity_of_articles_list(articles_list, n):
	perplexity_list = []
	for article in articles_list:
		perplexity_list.append(get_perplexity_of_article_ngram(article, n))
	return perplexity_list

def get_ngrams_for_article(sent, n):
	ngrams = []
	i = 0
	sent_tokens = sent.split()
	l = len(sent_tokens)
	while(i+n<=l):
    # for i in range(0,l-n):
		ngrams.append("__".join(sent_tokens[i:i+n]))
		i += 1
	return ngrams



articles = get_articles(TRAIN)
labels = get_labels(TRAIN_LABELS)
pos_articles, neg_articles = split_articles(articles, labels)

pos_perplexity3 = get_perplexity_of_articles_list(pos_articles,1)
neg_perplexity3 = get_perplexity_of_articles_list(neg_articles,1)

pos_perplexity4 = get_perplexity_of_articles_list(pos_articles,6)
neg_perplexity4 = get_perplexity_of_articles_list(neg_articles,6)

perp_fig = plt.figure()
x1 = range(1,len(pos_perplexity3)+1)
x2 = range(len(pos_perplexity3)+1, len(pos_perplexity3)+len(neg_perplexity3)+1)
# plt.scatter(x1, pos_perplexity, color='g')
# plt.scatter(x2, neg_perplexity, color='r')
plt.scatter(pos_perplexity3, pos_perplexity4, color='g', marker='+')
plt.scatter(neg_perplexity3, neg_perplexity4, color='r', marker='+')
plt.show()



