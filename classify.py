import numpy as np
from utils import *
from collections import Counter
import pdb
import math
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import eval
import parse
import sys

TRAIN = "../Data/trainingSet.dat"
TRAIN_LABELS = "../Data/trainingSetLabels.dat"

train_articles = get_articles(TRAIN)
train_labels = get_labels(TRAIN_LABELS)

train = np.loadtxt("Features/train_perp_wit_uncinc_out")
fname = 'perp_wit_uncinc_out'

dev_features_perplexity1 = np.array(parse.parse_file(parse.parse_indices_1gram_inclusive, "1gram/dev_"+fname))
dev_features_perplexity2 = np.array(parse.parse_file(parse.parse_indices_2gram_inclusive, "2gram/dev_"+fname))
dev_features_perplexity3 = np.array(parse.parse_file(parse.parse_indices_3gram_inclusive, "3gram/dev_"+fname))
dev_features_perplexity4 = np.array(parse.parse_file(parse.parse_indices_4gram_inclusive, "4gram/dev_"+fname))
dev_features_perplexity5 = np.array(parse.parse_file(parse.parse_indices_5gram_inclusive, "5gram/dev_"+fname))
dev = np.c_[dev_features_perplexity1, dev_features_perplexity2, dev_features_perplexity3, dev_features_perplexity4, dev_features_perplexity5]

scalar = StandardScaler()
train_features = scalar.fit_transform(train)
dev_features = scalar.transform(dev)

lg = LogisticRegression(C=0.1)
lg.fit(train_features, train_labels)


y_pred = lg.predict(dev_features)
y_pred_dev_proba = lg.predict_proba(dev_features)
y_pred_dev_proba1 = [y[1] for y in y_pred_dev_proba]
y_pred_dev_proba0 = [y[0] for y in y_pred_dev_proba]

results = []
for y0,y1,y in zip(y_pred_dev_proba0,y_pred_dev_proba1,y_pred):
	results.append(" ".join(map(str,[y0,y1,y])))

results_str = "\n".join(results)
print results_str
# print >> sys.stderr, results_str
