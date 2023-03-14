

import argparse
import sklearn.linear_model
import sklearn.feature_extraction
from collections import Counter
from sys import exit
import numpy as np
from scipy.sparse import coo_matrix, hstack

import os
from os.path import join as pjoin
import pickle
import sklearn.decomposition

parser = argparse.ArgumentParser(description='Baselines')

parser.add_argument("--corpus", type=str, default='books_5', help="books_5|books_old_5|books_8|books_all|gw_5|gw_8")
parser.add_argument("--outputdir", type=str, default='sandbox/', help="Output directory")
parser.add_argument("--features", type=str, default="bow", help="arora|bow|avg|mostcommonclass")
parser.add_argument("--ndims", type=int, default=1000)
parser.add_argument("--min_count", type=int, default=3)
parser.add_argument("--run_through_subset", type=bool, default=False)
parser.add_argument("--concat_sentences", type=bool, default=False)

params, _ = parser.parse_known_args()
print(params)

_PAD = "<pad>" # no need to pad
_UNK = "<unk>"
_START_VOCAB = [_PAD, _UNK]

def process_glove(glove_file, vocab_dict, save_path, random_init=True):
    """
    :param vocab_list: [vocab]
    :return:
    """
    if os.path.isfile(save_path + ".pkl"):
        print("Glove file already exists at %s" % (save_path + ".pkl"))
    else:
        glove_path = os.path.join(glove_file)
        if random_init:
            glove = {v: np.random.randn(300) for v in vocab_dict}
        else:
            glove = {v: np.zeros(300) for v in vocab_dict}

        found = 0

        for line in open(glove_path, "r"):
            word, vec = line.split(" ", 1)
            if word in vocab_dict:
                glove[word] = np.fromstring(vec, sep=" ")
                found += 1

        pickle.dump(glove, open(save_path + ".pkl", "wb"))

        print("saved glove data to: {}".format(save_path))
    return pickle.load(open(save_path + ".pkl", "rb"))

def create_vocabulary(vocabulary_path, corpus, discourse_markers=None):
	if os.path.isfile(vocabulary_path):
		print("Vocabulary file already exists at %s" % vocabulary_path)
	else:
		print("Creating vocabulary {}".format(vocabulary_path))

	vocab = {s:0 for s in _START_VOCAB}
	counter = 0

	for split in ["train", "test", "valid"]:
		split_data = corpus[split]
		for s_tag in ["s1", "s2"]:
			s_data = split_data[s_tag]
			for s in s_data:
				counter+=1
				if counter % 100000 == 0:
					print("processing line %d" % counter)
				for w in s.split(" "):
					if not w in _START_VOCAB:
						if w in vocab:
							vocab[w] += 1
						else:
							vocab[w] = 1

	vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
	print("Vocabulary size: %d" % len(vocab_list))
	with open(vocabulary_path, mode="w") as vocab_file:
		for w in vocab_list:
			vocab_file.write("{}\t{}\n".format(w, vocab[w]))

def initialize_vocabulary(vocabulary_path):
	# map vocab to word embeddings
	if os.path.isfile(vocabulary_path):
		counts = {}
		total = 0
		rev_vocab = []
		for line in open(vocabulary_path, mode="r"):
			s, c = line[:-1].split('\t')
			c = int(c)
			counts[s] = c
			total += c
			rev_vocab.append(s)
		vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
		return vocab, rev_vocab, counts, total
	else:
		raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def unigrams_phi(words):
	return Counter(words)

def bigrams_phi(words):
	return Counter([words[i-2:i] for i in range(len(words)) if i>1])

def trigrams_phi(words):
	return Counter([words[i-3:i] for i in range(len(words)) if i>3])

def unigrams(words):
	return words

def bigrams(words):
	return [" ".join(words[i-2:i]) for i in range(len(words)) if i>1]

def trigrams(words):
	return [" ".join(words[i-3:i]) for i in range(len(words)) if i>3]

def ngrams(sentence):
	words = sentence.split(" ")
	return trigrams(words) + bigrams(words) + unigrams(words)

def ngrams_phi(words):
	return Counter(trigrams(words) + bigrams(words) + unigrams(words))

def read_corpus(corpus_label):
	corpus_dir = "/home/anie/DisExtract/data/books/"
	labels = {
		"books_all": "discourse_EN_ALL_and_then_because_though_still_after_when_while_but_also_as_so_although_before_if_2017dec21_",
		"books_8": "discourse_EN_EIGHT_and_but_because_if_when_before_so_though_2017dec18_",
		"books_5": "discourse_EN_FIVE_and_but_because_if_when_2017dec12_"
	}
	corpus = {}
	for split in ["train", "valid", "test"]:
		filename = corpus_dir + labels[corpus_label] + split + ".tsv"
		print("reading {}".format(filename))
		corpus[split] = {"s1": [], "s2": [], "label": []}
		for line in open(filename):
			s1, s2, label = line[:-1].split("\t")
			corpus[split]["s1"].append(s1)
			corpus[split]["s2"].append(s2)
			corpus[split]["label"].append(label)
	return corpus

def get_most_common_class_accuracies():
	corpus = read_corpus(params.corpus)
	train_counter = Counter(corpus["train"]["label"])
	print(train_counter)
	test_counter = Counter(corpus["test"]["label"])
	print(test_counter)
	most_common_train_class = train_counter.most_common()[0][0]
	n_correct = test_counter[most_common_train_class]
	n_total = sum([test_counter[label] for label in test_counter])
	print(float(n_correct)/n_total)

def get_features_combined(params):
	corpus = read_corpus(params.corpus)
	if params.run_through_subset:
		corpus["train"]["s1"] = corpus["train"]["s1"][:100]
		corpus["train"]["s2"] = corpus["train"]["s2"][:100]
		corpus["train"]["label"] = corpus["train"]["label"][:100]
		corpus["test"]["s1"] = corpus["test"]["s1"][:100]
		corpus["test"]["s2"] = corpus["test"]["s2"][:100]
		corpus["test"]["label"] = corpus["test"]["label"][:100]
	vectorizer = sklearn.feature_extraction.DictVectorizer()
	train_dicts = []
	test_dicts = []
	train_labels = []
	test_labels = []
	all_features = Counter([])
	print("collecting combined ngrams")
	for i in range(len(corpus["train"]["s1"])):
		s1 = corpus["train"]["s1"][i]
		s2 = corpus["train"]["s2"][i]
		label = corpus["train"]["label"][i]
		features = ngrams(s1) + ngrams(s2)
		all_features.update(features)
		train_labels.append(label)
		train_dicts.append(Counter(features))
		if i%100000==0:
                        print("{}K of {}K in training set processed".format(round(i/1000), round(len(corpus["train"]["s1"])/1000)))
	for i in range(len(corpus["test"]["s1"])):
		s1 = corpus["test"]["s1"][i]
		s2 = corpus["test"]["s2"][i]
		label = corpus["test"]["label"][i]
		features = ngrams(s1) + ngrams(s2)
		all_features.update(features)
		test_labels.append(label)
		test_dicts.append(Counter(features))
		if i%100000==0:
			print("{}K of {}K in training set processed".format(round(i/1000), round(len(corpus["test"]["s1"])/1000)))
	all_dicts = train_dicts + test_dicts
	support = [feature for feature in all_features if all_features[feature]>=params.min_count]
	print("vectorizing dataset")
	feat_matrix = vectorizer.fit(all_dicts)
	vectorizer.restrict(support)
	train = vectorizer.transform(train_dicts)
	test = vectorizer.transform(test_dicts)
	return (train, test, train_labels, test_labels)


def get_features(params):
	corpus = read_corpus(params.corpus)
	if params.run_through_subset:
		corpus["train"]["s1"] = corpus["train"]["s1"][:100]
		corpus["train"]["s2"] = corpus["train"]["s2"][:100]
		corpus["train"]["label"] = corpus["train"]["label"][:100]
		corpus["test"]["s1"] = corpus["test"]["s1"][:100]
		corpus["test"]["s2"] = corpus["test"]["s2"][:100]
		corpus["test"]["label"] = corpus["test"]["label"][:100]
	vectorizer = sklearn.feature_extraction.DictVectorizer()
	train_dicts_1 = []
	train_dicts_2 = []
	test_dicts_1 = []
	test_dicts_2 = []
	train_labels = []
	test_labels = []
	all_features = Counter([])
	print("collecting ngrams")
	for i in range(len(corpus["train"]["s1"])):
		s1 = corpus["train"]["s1"][i]
		s2 = corpus["train"]["s2"][i]
		features1 = ngrams(s1)
		features2 = ngrams(s2)
		all_features.update(features1 + features2)
		label = corpus["train"]["label"][i]
		train_labels.append(label)
		train_dicts_1.append(Counter(features1))
		train_dicts_2.append(Counter(features2))
		if i%100000==0:
			print("{}K of {}K in training set processed".format(round(i/1000), round(len(corpus["train"]["s1"])/1000)))
	for i in range(len(corpus["test"]["s1"])):
		s1 = corpus["test"]["s1"][i]
		s2 = corpus["test"]["s2"][i]
		features1 = ngrams(s1)
		features2 = ngrams(s2)
		all_features.update(features1 + features2)
		label = corpus["test"]["label"][i]
		test_labels.append(label)
		test_dicts_1.append(Counter(features1))
		test_dicts_2.append(Counter(features2))
		if i%100000==0:
			print("{}K of {}K in test set processed".format(round(i/1000), round(len(corpus["test"]["s1"])/1000)))
	all_dicts = train_dicts_1 + train_dicts_2 + test_dicts_1 + test_dicts_2
	support = [feature for feature in all_features if all_features[feature]>=params.min_count]
	print("vectorizing dataset")
	feat_matrix = vectorizer.fit(all_dicts)
	vectorizer.restrict(support)
	train1 = vectorizer.transform(train_dicts_1)
	train2 = vectorizer.transform(train_dicts_2)
	test1 = vectorizer.transform(test_dicts_1)
	test2 = vectorizer.transform(test_dicts_2)
	return (train1, train2, test1, test2, train_labels, test_labels)

def aggregate_features(params):
	train1, train2, test1, test2, train_labels, test_labels = get_features(params)
	return (hstack([train1, train2]).toarray(), hstack([test1, test2]).toarray(), train_labels, test_labels)

def run_baseline_BoW_model(params):
	if params.concat_sentences:
		train_X, test_X, train_y, test_y = aggregate_features(params)
	else:
		train_X, test_X, train_y, test_y = get_features_combined(params)
	print(train_X.shape)
	print(test_X.shape)
	print("fitting model")
	model = sklearn.linear_model.LogisticRegression(max_iter=100, verbose=0, fit_intercept=True)
	model.fit(train_X, train_y)
	print("testing model")
	test_pred = model.predict(test_X)
	results = {label: {"hits": 0, "actual": 0.0000001, "predicted": 0.00000001} for label in set(train_y)}
	for i in range(len(test_y)):
		actual = test_y[i]
		predicted = test_pred[i]
		if actual==predicted:
			results[actual]["hits"] += 1
		results[actual]["actual"] += 1
		results[predicted]["predicted"] += 1
	for label in results:
		precision = 100*float(results[label]["hits"]) / results[label]["predicted"]
		recall = 100*float(results[label]["hits"]) / results[label]["actual"]
		print("{}: precision={:.2f}, recall={:.2f}".format(label, precision, recall))
	output = model.score(test_X, test_y)
	print("accuracy: {}".format(output))
	print("accuracy: {}".format(np.sum(test_pred==test_y)/len(test_pred)))
	np.save("predicted.txt", test_pred)
	np.save("actual.txt", test_y)

def get_X(lst, glove_dict, counts, total, a):
	X = []
	for s in lst:
		words = s.split(" ")
		weighted_word_embeddings = []
		for word in words:
			vw = glove_dict[word]
			pw = counts[word]/total
			weighted_word_embeddings.append(a/(a+pw)*vw)
		vs = 1.0/len(words) * np.sum(weighted_word_embeddings, axis=0)
		X.append(vs)
	return np.matrix(X)

def run_baseline_arora_model(params):
	glove_file = "/home/anie/glove/glove.840B.300d.txt"
	vocab_path = pjoin(params.outputdir, params.corpus + "_vocab.dat")
	corpus = read_corpus(params.corpus)
	if params.run_through_subset:
		corpus["train"]["s1"] = corpus["train"]["s1"][:100]
		corpus["train"]["s2"] = corpus["train"]["s2"][:100]
		corpus["train"]["label"] = corpus["train"]["label"][:100]
		corpus["test"]["s1"] = corpus["test"]["s1"][:100]
		corpus["test"]["s2"] = corpus["test"]["s2"][:100]
		corpus["test"]["label"] = corpus["test"]["label"][:100]

	if not os.path.exists(vocab_path):
		create_vocabulary(vocab_path, corpus)
	vocab, rev_vocab, counts, total = initialize_vocabulary(vocab_path)
	# get word vectors
	glove_dict = process_glove(glove_file, vocab, pjoin(params.outputdir, params.corpus + "_glove"))
	a = 0.001 ## search for this tune on validation set a=[]

	print("processing corpus")
	X_train1 = get_X(corpus["train"]["s1"], glove_dict, counts, total, a)
	X_train2 = get_X(corpus["train"]["s2"], glove_dict, counts, total, a)
	X_valid1 = get_X(corpus["valid"]["s1"], glove_dict, counts, total, a) 
	X_valid2 = get_X(corpus["valid"]["s2"], glove_dict, counts, total, a)
	X_test1 = get_X(corpus["test"]["s1"], glove_dict, counts, total, a) 
	X_test2 = get_X(corpus["test"]["s2"], glove_dict, counts, total, a)
	X = np.concatenate((X_train1, X_train2), axis=0) # , X_valid1, X_valid2, X_test1, X_test2
	pca = sklearn.decomposition.PCA(n_components=1)
	pca.fit(X)
	u = pca.components_
	# projection_matrix = np.matmul(np.transpose(u), u)
	projection_matrix = np.outer(u, u) # or transpose u

	print("getting train and test sets")
	# X_train1 = X_train1 - np.transpose(np.matmul(projection_matrix, np.transpose(X_train1)))
	# X_train2 = X_train2 - np.transpose(np.matmul(projection_matrix, np.transpose(X_train2)))
	X_train1 = X_train1 - X_train1.dot(u.transpose()).dot(u)
	X_train2 = X_train2 - X_train2.dot(u.transpose()).dot(u)

	## do all the vector functions :)
	train_X = np.concatenate((X_train1, X_train2, X_train1 + X_train2, X_train1 - X_train2, X_train1 * X_train2 ), axis=1)
	train_y = corpus["train"]["label"]

	X_test1 = X_test1 - np.transpose(np.matmul(projection_matrix, np.transpose(X_test1)))
	X_test2 = X_test2 - np.transpose(np.matmul(projection_matrix, np.transpose(X_test2)))
	test_X = np.concatenate((X_test1, X_test2, X_test1 + X_test2, X_test1 - X_test2, X_test1 * X_test2), axis=1)
	test_y = corpus["test"]["label"]

	# get 1st PC of X_train --> u
	# rewrite v_s
	# run linear regression
	print("fitting model")

	# Try MLPClassifier in sklearn
	model = sklearn.linear_model.LogisticRegression(max_iter=1000, verbose=0, fit_intercept=True)
	model.fit(train_X, train_y)
	print("testing model")
	test_pred = model.predict(test_X)
	results = {label: {"hits": 0, "actual": 0.0000001, "predicted": 0.00000001} for label in set(train_y)}
	for i in range(len(test_y)):
		actual = test_y[i]
		predicted = test_pred[i]
		if actual==predicted:
			results[actual]["hits"] += 1
		results[actual]["actual"] += 1
		results[predicted]["predicted"] += 1
	for label in results:
		precision = 100*float(results[label]["hits"]) / results[label]["predicted"]
		recall = 100*float(results[label]["hits"]) / results[label]["actual"]
		print("{}: precision={:.2f}, recall={:.2f}".format(label, precision, recall))
	output = model.score(test_X, test_y)
	print("accuracy: {}".format(output))
	print("accuracy: {}".format(np.sum(test_pred==test_y)/len(test_pred)))
	np.save("predicted.txt", test_pred)
	np.save("actual.txt", test_y)

if __name__ == '__main__':
	if params.features == "mostcommonclass":
		get_most_common_class_accuracies()
	elif params.features == "bow":
		run_baseline_BoW_model(params)
	elif params.features == "arora":
		run_baseline_arora_model(params)

