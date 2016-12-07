import numpy as np
import utils
import matplotlib.pyplot as plt
import pdb

WORD2VEC_FNAME = "../Data/glove.6B/glove.6B.300d.txt"
GOOGLE_WORD2VEC_FNAME = '/Users/ksathyen/Documents/CMU_MLT_Course_Material/Sem_3/LAS/Project/Data/GoogleNews-vectors-negative300.bin'
def load_word2vec(fname, google=False):
	if not google:
		word2vec = {}
		with open(fname) as f:
			for line in f:
				line_tokens = line.rstrip("\n").split()
				word, vect_list = line_tokens[0],line_tokens[1:]
				vect_list_float = map(float, vect_list)
				word2vec[word] = np.array(vect_list_float)
			return word2vec
	else:
		import gensim
		word2vecmodel = gensim.models.Word2Vec.load_word2vec_format(GOOGLE_WORD2VEC_FNAME, binary=True)
		return word2vecmodel


word2vec = load_word2vec(WORD2VEC_FNAME)
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

def get_w2vfeature(article):
	article = " ".join(article)
	article = article.split()
	vect_sum = np.zeros((1,300))
	num_words = 0
	vect_list = []
	for word in article:
		# pdb.set_trace()
		try:
		# if 1:
			print word.lower()
			# pdb.set_trace()
			if word.lower() not in stopwords:
				vect = word2vec[word.lower()]
				vect_list.append(vect)
				num_words += 1
				vect_sum = vect_sum + vect
		# else:
		except:
			continue
	# print num_words
	if num_words == 0:
		return 0
	# pdb.set_trace()
	vect_centroid = (1.0/num_words)*(vect_sum[0])
	vects_diff = np.array(vect_list) - vect_centroid
	vect_dist = np.power(vects_diff, 2).sum(axis=1)
	# pdb.set_trace()
	vect_dist = vect_dist.sum()/(1.0*num_words)
	print vect_dist
	return vect_dist

def get_w2vfeatures_list(article_list):
	w2vfeatures = []
	i = 0
	for article in article_list:
		print i
		i += 1
		w2vfeatures.append(get_w2vfeature(article))
	w2vfeatures = np.array(w2vfeatures).T
	return w2vfeatures

train_articles = utils.get_articles(utils.TRAIN)
train_labels = utils.get_labels(utils.TRAIN_LABELS)
train_w2vfeatures = get_w2vfeatures_list(train_articles)
np.savetxt("Features/google_word2vec_train.txt",train_features)
# train_w2vfeatures = np.loadtxt("Features/google_word2vec_train.txt")

# dev_articles = utils.get_articles(utils.DEV)
# dev_labels = utils.get_labels(utils.DEV_LABELS)
# dev_features = get_w2vfeatures_list(dev_articles)
# np.savetxt("Features/google_word2vec_dev.txt",dev_features)


# train_w2vfeatures = np.array(train_w2vfeatures).reshape(1,1000)

plt.figure()
plt.scatter(range(0,len(train_w2vfeatures)) ,train_w2vfeatures, c=train_labels, cmap=plt.cm.coolwarm)
plt.show()


