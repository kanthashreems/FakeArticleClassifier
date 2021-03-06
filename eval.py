import numpy as np
import sklearn.metrics
import sys
import utils

TRAIN_TRUE = "../Data/trainingSetLabels.dat"
DEV_TRUE = "../Data/developmentSetLabels.dat"

train_true = np.array(utils.get_labels(TRAIN_TRUE))
dev_true = np.array(utils.get_labels(DEV_TRUE))

def classification_error(y_pred, dev=0):
	if dev:
		y_true = dev_true
	else:
		y_true = train_true
	return calculate_classification_error(y_true, y_pred, script=1)	

def soft_metric(y0_prob, y1_prob, dev=0):
	if dev:
		y_true = dev_true
	else:
		y_true = train_true
	return calculate_soft_score(y_true, y0_prob, y1_prob, script=1)

def calculate_classification_error(y_true, y_pred, script=0):
	hard_metric = sklearn.metrics.accuracy_score(y_true, y_pred)
	if script:
		print "Hard Metric: " + str(hard_metric)
	return hard_metric

def calculate_soft_score(y_true, y0_prob,y1_prob,script=0):
	log_posterior = np.multiply(y_true, np.log2(y1_prob)) + np.multiply(1-y_true, np.log2(y0_prob))
	avg_log_posterior = log_posterior.sum()*(1.0/y0_prob.size)
	if script:
		print "Soft Metric:" + str(avg_log_posterior)
	return avg_log_posterior

def get_probabilities(pred_file):
	with open(pred_file, "r") as inp:
		labels = inp.read().rstrip("\n")
	labels = labels.split("\n")
	y0_prob = []
	y1_prob = []
	y_pred = []
	for label in labels:
		y0,y1,y = label.split()
		y0_prob.append(y0)
		y1_prob.append(y1)
		y_pred.append(y)
	y0_prob = map(float, y0_prob)
	y1_prob = map(float, y1_prob)
	y_pred = map(float, y_pred)
	return np.array(y0_prob), np.array(y1_prob), np.array(y_pred)

if __name__ == "__main__":
	true_file = sys.argv[1]
	pred_file = sys.argv[2]
	y0_prob, y1_prob, y_pred = get_probabilities(pred_file)
	y_true = np.array(utils.get_labels(true_file))
	calculate_classification_error(y_true, y_pred, 1)
	calculate_soft_score(y_true, y0_prob, y1_prob, 1)



