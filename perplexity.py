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
import degenerateEM
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingClassifier


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
train_features_entropy = np.loadtxt("Features/train_perplexity_f.txt")
dev_features_entropy = np.loadtxt("Features/dev_perplexity_f.txt")
# np.savetxt("train_perplexity_f.txt", train_features)

# np.savetxt("train_perplexity_f.txt", train_features)
scalar = StandardScaler()
fname = "perp_wit_uncinc_out" #best

# fname = "perp_lin_out" 

train_features_perplexity1 = np.array(parse.parse_file(parse.parse_indices_1gram_inclusive, "1gram/"+fname))
train_features_perplexity2 = np.array(parse.parse_file(parse.parse_indices_2gram_inclusive, "2gram/"+fname))
train_features_perplexity3 = np.array(parse.parse_file(parse.parse_indices_3gram_inclusive, "3gram/"+fname))
train_features_perplexity4 = np.array(parse.parse_file(parse.parse_indices_4gram_inclusive, "4gram/"+fname))
train_features_perplexity5 = np.array(parse.parse_file(parse.parse_indices_5gram_inclusive, "5gram/"+fname))
train= np.c_[train_features_perplexity1, train_features_perplexity2, train_features_perplexity3, train_features_perplexity4, train_features_perplexity5]
# np.savetxt("Features/train_"+fname, train_features1)

# fname = "pos_perp_lin_out" 
# # train_features_perplexity1 = np.array(parse.parse_file(parse.parse_indices_1gram_inclusive, "1gram/"+fname))
# train_features_perplexity2 = np.array(parse.parse_file(parse.parse_indices_2gram, "2gram/"+fname))
# train_features_perplexity3 = np.array(parse.parse_file(parse.parse_indices_3gram, "3gram/"+fname))
# train_features_perplexity4 = np.array(parse.parse_file(parse.parse_indices_4gram, "4gram/"+fname))
# train_features_perplexity5 = np.array(parse.parse_file(parse.parse_indices_5gram, "5gram/"+fname))
# train_features2 = np.c_[train_features_perplexity2, train_features_perplexity3, train_features_perplexity4, train_features_perplexity5]
# train_features = np.c_[train_features1, train_features2]

# np.savetxt("Features/train_"+fname, train_features2)


# # train_w2v = np.loadtxt("Features/google_word2vec_train.txt")
# # train_features = np.c_[train_features_perplexity1, train_features_perplexity2, train_features_perplexity3, train_features_perplexity4, train_features_perplexity5]
# train_features = scalar.fit_transform(train_features)
dev_articles = get_articles(DEV)
dev_labels = get_labels(DEV_LABELS)
# # dev_features = get_perplexity_features(dev_articles)
# # dev_features = np.loadtxt("Features/dev_perplexity_f.txt")
# # dev_features = scalar.transform(dev_features)
# fname = "perp_lin_out" 
dev_features_perplexity1 = np.array(parse.parse_file(parse.parse_indices_1gram_inclusive, "1gram/dev_"+fname))
dev_features_perplexity2 = np.array(parse.parse_file(parse.parse_indices_2gram_inclusive, "2gram/dev_"+fname))
dev_features_perplexity3 = np.array(parse.parse_file(parse.parse_indices_3gram_inclusive, "3gram/dev_"+fname))
dev_features_perplexity4 = np.array(parse.parse_file(parse.parse_indices_4gram_inclusive, "4gram/dev_"+fname))
dev_features_perplexity5 = np.array(parse.parse_file(parse.parse_indices_5gram_inclusive, "5gram/dev_"+fname))
# # dev_w2v = np.loadtxt("Features/google_word2vec_dev.txt")
dev = np.c_[dev_features_perplexity1, dev_features_perplexity2, dev_features_perplexity3, dev_features_perplexity4, dev_features_perplexity5]
# np.savetxt("Features/dev_"+fname, dev_features1)


# fname = "pos_perp_lin_out" 
# # dev_features_perplexity1 = np.array(parse.parse_file(parse.parse_indices_1gram_inclusive, "1gram/dev_"+fname))
# dev_features_perplexity2 = np.array(parse.parse_file(parse.parse_indices_2gram, "2gram/dev_"+fname))
# dev_features_perplexity3 = np.array(parse.parse_file(parse.parse_indices_3gram, "3gram/dev_"+fname))
# dev_features_perplexity4 = np.array(parse.parse_file(parse.parse_indices_4gram, "4gram/dev_"+fname))
# dev_features_perplexity5 = np.array(parse.parse_file(parse.parse_indices_5gram, "5gram/dev_"+fname))
# # dev_w2v = np.loadtxt("Features/google_word2vec_dev.txt")
# dev_features2 = np.c_[dev_features_perplexity2, dev_features_perplexity3, dev_features_perplexity4, dev_features_perplexity5]
# dev_features = np.c_[dev_features1,dev_features2]
# np.savetxt("Features/dev_"+fname, dev_features2)


# train=None
# dev = None
# fname_list = ['perp_lin_out','perp_wit_out','perp_lin_uncinc_out','perp_lin_uncexc_out','perp_wit_uncinc_out','perp_wit_uncexc_out','pos_perp_lin_out','pos_perp_wit_out','pos_perp_lin_uncinc_out','pos_perp_lin_uncexc_out','pos_perp_wit_uncinc_out','pos_perp_wit_uncexc_out']
# for fname in ['perp_wit_uncinc_out', 'pos_perp_wit_uncinc_out']:
# 	train_features = np.loadtxt("Features/train_"+fname)
# 	dev_features = np.loadtxt("Features/dev_"+fname)

# 	if train == None:
# 		train = np.c_[train_features]
# 		dev = np.c_[dev_features]
# 	else:
# 		train = np.c_[train,train_features]
# 		dev = np.c_[dev,dev_features]



train_features = scalar.fit_transform(train)

dev_features = scalar.transform(dev)
# X = train_features
# n1 = 0
# n2 = 1
# x_min, x_max = X[:, n1].min() - 1, X[:, n1].max() + 1
# y_min, y_max = X[:, n2].min() - 1, X[:, n2].max() + 1
# h = 0.2

# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
for g in np.arange(0.1,1,0.1):
	print g
	# lg = SVC(kernel='rbf', gamma = 0.0001, C=0.5)
	print dev_features.shape
	lg1 = LogisticRegression(C=g)
	# lg1 = AdaBoostRegressor(base_estimator=LogisticRegression(), n_estimators=g)
	lg1.fit(train_features, train_labels)
	print train_features.shape


	# np.savetxt("dev_perplexity_f.txt", dev_features)

	y_pred_train = lg1.predict(train_features)
	y_pred = lg1.predict(dev_features)

	y_pred_train_proba = lg1.predict_proba(train_features)
	y_pred_train_proba1 = [y[1] for y in y_pred_train_proba]
	y_pred_train_proba0 = [y[0] for y in y_pred_train_proba]

	y_pred_dev_proba = lg1.predict_proba(dev_features)
	y_pred_dev_proba1 = [y[1] for y in y_pred_dev_proba]
	y_pred_dev_proba0 = [y[0] for y in y_pred_dev_proba]

	# lg2 = LogisticRegression(C=1.5)
	# lg2.fit(train_features_entropy,train_labels)

	# eclf = VotingClassifier(estimators=[('lr', lg1), ('lr', lg2)], voting='soft')
	# eclf.fit(train_features, train_labels)
	# y_pred_train = eclf.predict(train_features)
	# y_pred = eclf.predict(dev_features)

	# y_pred_train_proba = eclf.predict_proba(train_features)
	# y_pred_train_proba1 = [y[1] for y in y_pred_train_proba]
	# y_pred_train_proba0 = [y[0] for y in y_pred_train_proba]

	# y_pred_dev_proba = eclf.predict_proba(dev_features)
	# y_pred_dev_proba1 = [y[1] for y in y_pred_dev_proba]
	# y_pred_dev_proba0 = [y[0] for y in y_pred_dev_proba]



	# y_train_entropy = lg2.predict(train_features_entropy)
	# y_train_entropy_proba = lg2.predict_proba(train_features_entropy)
	# y_train_entropy_proba1 = [y[1] for y in y_train_entropy_proba]
	# y_dev_entropy = lg2.predict(dev_features_entropy)
	# y_dev_entropy_proba = lg2.predict_proba(dev_features_entropy)
	# y_dev_entropy_proba1 = [y[1] for y in y_dev_entropy_proba]

	# streams = np.array([y_pred_train_proba1, y_train_entropy_proba1]).transpose()
	# lamda = np.array([1,0])
	# # lamda = np.array([1.0/3,1.0/3,1.0/3])
	# # lamda = degenerateEM.degenerateEM(streams, 0.00001)
	# print lamda
	# l_old = lamda
	# l_diag = np.diag(l_old)
	# # log_likelihood_old = log_likelihood

	# lamda = np.array([1,0])
	# weighted_models = np.dot(streams, l_diag).sum(axis=1) 
	
	# y_pred_train = (weighted_models>0.5).astype(int)

	# streams = np.array([y_pred_dev_proba1, y_dev_entropy_proba1]).transpose()
	# lamda = np.array([1,0])
	# weighted_models = np.dot(streams, l_diag).sum(axis=1) 

	

	# y_pred = (weighted_models>0.5).astype(int)


	print "Train:"
	eval.classification_error(y_pred_train, dev=0)
	print "Dev:"
	eval.classification_error(y_pred, dev=1)

	print "Train:"
	eval.soft_metric(np.array(y_pred_train_proba0), np.array(y_pred_train_proba1) , dev=0)
	print "Dev:"
	eval.soft_metric(np.array(y_pred_dev_proba0), np.array(y_pred_dev_proba1), dev=1)

	# print "entropy"
	# print "Train:"
	# eval.classification_error(y_train_entropy, dev=0)
	# print "Dev:"
	# eval.classification_error(y_dev_entropy, dev=1)
	print "####################"


	
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

	# false pos and false neg
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





