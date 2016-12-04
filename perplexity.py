import numpy as np
from utils import *
from collections import Counter
import matplotlib.pyplot as plt
import pdb
import math
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import eval
import parse


# def get_entropy(count_list):
#     count_arr = np.array(count_list)
#     s = sum(count_arr)
#     if s==0:
#         return 1
#     prob_arr = (1.0/s)*count_arr
#     return -sum(np.multiply(prob_arr, np.log2(prob_arr)))
stemmer = SnowballStemmer("english")

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

def stem(word):
	stemmed = word
	if word not in ["<UNK>","<stop>"]:
		stemmed = stemmer.stem(word)
		# print stemmed, word
	return stemmed

def stem_sentence(sent):
	stemmed_sent = map(stem, sent.split())
	return " ".join(stemmed_sent)

def get_perplexity_of_article_ngram(article, n):
	article = [s.replace("<s>","").replace("</s>","") for s in article]
	# stemmed = [stemmer.stem(meaningful_word) for meaningful_word in meaningful_words]
	article = map(stem_sentence, article)
	# print article
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

def get_perplexity_features(articles):
	perplexity1 = get_perplexity_of_articles_list(articles,1)
	print "1"
	perplexity2 = get_perplexity_of_articles_list(articles,2)
	print "2"
	# perplexity3 = get_perplexity_of_articles_list(articles,3)
	print "3"
	# perplexity4 = get_perplexity_of_articles_list(articles,4)
	print "4"
	# perplexity5 = get_perplexity_of_articles_list(articles,5)
	print "5"
	features = np.array([perplexity1,perplexity2]).T
	return features


train_articles = get_articles(TRAIN)
train_labels = get_labels(TRAIN_LABELS)
# train_features = get_perplexity_features(train_articles)
train_features = np.loadtxt("Features/train_perplexity_f.txt")
# np.savetxt("train_perplexity_f.txt", train_features)

np.savetxt("train_perplexity_f.txt", train_features)
scalar = StandardScaler()

train_features_perplexity4 = np.array(parse.parse_file(parse.parse_indices_4gram, "4gram/perp_wit_out"))[:,0]
train_features_perplexity3 = np.array(parse.parse_file(parse.parse_indices_3gram, "3gram/perp_wit_out"))[:,0]
train_w2v = np.loadtxt("Features/word2vec_train.txt")
train_features = np.c_[train_features_perplexity3, train_features_perplexity4, train_w2v]
# train_features = scalar.fit_transform(train_features)
dev_articles = get_articles(DEV)
dev_labels = get_labels(DEV_LABELS)
# dev_features = get_perplexity_features(dev_articles)
dev_features = np.loadtxt("Features/dev_perplexity_f.txt")
# dev_features = scalar.transform(dev_features)
dev_features_perplexity3 = np.array(parse.parse_file(parse.parse_indices_3gram, "3gram/dev_perp_wit_out"))[:,0]
dev_features_perplexity4 = np.array(parse.parse_file(parse.parse_indices_4gram, "4gram/dev_perp_wit_out"))[:,0]
dev_w2v = np.loadtxt("Features/word2vec_dev.txt")
dev_features = np.c_[dev_features_perplexity3, dev_features_perplexity4, dev_w2v]
# dev_features = scalar.transform(dev_features)
X = train_features
n1 = 0
n2 = 1
x_min, x_max = X[:, n1].min() - 1, X[:, n1].max() + 1
y_min, y_max = X[:, n2].min() - 1, X[:, n2].max() + 1
h = 0.2

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
for g in [0]:
	# lg = SVC(kernel='linear', gamma = 0.0001, C=0.5)
	lg = LogisticRegression()
	lg.fit(train_features[:,[2]], train_labels)


	# np.savetxt("dev_perplexity_f.txt", dev_features)

	y_pred_train = lg.predict(train_features[:,[2]])
	y_pred = lg.predict(dev_features[:,[2]])

	print "Train:"
	eval.classification_error(y_pred_train, dev=0)
	print "Dev:"
	eval.classification_error(y_pred, dev=1)


	
# 	print X.shape
	
# 	Z = lg.predict(np.c_[xx.ravel(), yy.ravel()])

# # 	# Put the result into a color plot
# 	Z = Z.reshape(xx.shape)
# 	plt.figure(figsize=(5,5))
# 	plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# # 	# # Plot also the training points
	
# 	plt.scatter(X[:, 0], X[:, 1], c=train_labels, cmap=plt.cm.coolwarm, marker = "+")
# 	plt.scatter(dev_features[:, 0], dev_features[:, 1], c=dev_labels, cmap=plt.cm.coolwarm)
# 	plt.xlabel('1 gram')
# 	plt.ylabel('2 gram')
# 	plt.xlim(xx.min(), xx.max())
# 	plt.ylim(yy.min(), yy.max())
# 	# plt.xticks(())
# 	# plt.yticks(())
# 	plt.title("Decision boundary C:" + str(g))

	#false pos and false neg
	y_diff = y_pred - np.array(dev_labels)
	# pdb.set_trace()
	y_fp_idx = (y_diff == 1)
	y_fp = list(np.array(dev_articles)[y_fp_idx])
	y_fn_idx = (y_diff == -1)
	y_fn = list(np.array(dev_articles)[y_fn_idx])

	y_fp_articles = "\n~~~~~ \n".join(["\n".join(a) for a in y_fp])
	y_fn_articles = "\n~~~~~ \n".join(["\n".join(a) for a in y_fn])
	write("fp.txt", y_fp_articles)
	write("fn.txt", y_fn_articles)


plt.show()





# perp_fig = plt.figure()
# x1 = range(1,len(pos_perplexity3)+1)
# x2 = range(len(pos_perplexity3)+1, len(pos_perplexity3)+len(neg_perplexity3)+1)
# # plt.scatter(x1, pos_perplexity, color='g')
# # plt.scatter(x2, neg_perplexity, color='r')
# plt.scatter(pos_perplexity3, pos_perplexity4, color='g', marker='+')
# plt.scatter(neg_perplexity3, neg_perplexity4, color='r', marker='+')
# plt.show()





