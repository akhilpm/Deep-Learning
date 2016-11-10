import nltk
import math
import time
import numpy as np
import tensorflow as tf
import pickle
import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from nltk.util import ngrams
from datetime import datetime
from tflearn.data_utils import VocabularyProcessor


def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()


def build_vocabulary(corpus_file, skip_window):

	data = list(open(corpus_file, 'r').readlines())
	data = [s.strip() for s in data]
	data = [clean_str(sent) for sent in data]
	#max_doc_length = max([len(x.split(' ')) for x in data])
	max_doc_length = 100

	VocabProcessor = VocabularyProcessor(max_doc_length, min_frequency=3)
	mat = VocabProcessor.fit_transform(data)
	#mat = np.array(list(mat))

	print('Vocabulary size : %d' %len(VocabProcessor.vocabulary_))
	VocabProcessor.save('vocabulary.txt')
	return VocabProcessor, max_doc_length


def get_word_indices(VocabProcessor, words, no_of_words):
	""" for skip-gram model """
	if len(words)==1:
		indices = VocabProcessor.transform(words)
		indices = np.array(list(indices)).flatten()[0]

		temp = np.ones(no_of_words, dtype=np.int32)
		return indices*temp

	else:
		indices = VocabProcessor.transform(words)
		indices = np.array(list(indices))
		return indices[:,0]	



def generate_batch(iterator, flags, VocabProcessor):
	""" gets a batch of training data """

	batch_size = flags.FLAGS.batch_size
	skip_window = flags.FLAGS.skip_window
	no_of_ngrams = batch_size/(2*skip_window)
	no_of_words = 2*skip_window

	x_list = np.array([], dtype=np.int32)
	y_list = np.array([], dtype=np.int32)

	for i in xrange(no_of_ngrams):
		try:
			text = iterator.next()

		except StopIteration:
			datafile = open(flags.FLAGS.corpus)
			content = datafile.read().replace('\n', ' ')
			iterator = ngrams(content.split(), flags.FLAGS.n)
			text = iterator.next()

		text = list(text)
		source = text[skip_window]
		#print(type(source))
		text.remove(source)

		x_indices = get_word_indices(VocabProcessor, text, no_of_words)
		y_indices = get_word_indices(VocabProcessor, [source], no_of_words)
		#print(x_indices)
		#print(y_indices)

		x_list = np.append(x_list, x_indices)
		y_list = np.append(y_list, y_indices)
		

	#print('y_list length %d' %len(y_list))
	#print('x_list length %d' %len(x_list))
	y_list = y_list.reshape(batch_size, 1)
	return iterator, x_list, y_list			



def set_flags(vocabulary_size, skip_window, corpus):
	flags = tf.app.flags
	
	flags.DEFINE_float('eta', 0.1, 'Learning rate.')
	flags.DEFINE_integer('max_steps', 30001, 'Number of steps to run trainer.')
	flags.DEFINE_integer('embedd_size', 100, 'Word embedding size')
	flags.DEFINE_integer('skip_window', skip_window, 'Range of context words for skip-gram model')
	flags.DEFINE_integer('batch_size', 20, 'Batch size, Must divide evenly into the dataset sizes.')
	flags.DEFINE_string('corpus', corpus, 'Corpus used to train the model.')
	flags.DEFINE_integer('vocabulary_size', vocabulary_size, 'No. of distinct words in the corpus')
	flags.DEFINE_integer('neg_samples', 100, 'No. of negative samples to consider')
	flags.DEFINE_integer('n', 2*skip_window+1, 'n value for the n-gram in nltk')

	#accessing eg: flags.FLAGS.max_steps
	return flags


def plot_embedding(embedding, VocabProcessor):
	""" plots the produced embedding using TSNE """

	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

	plot_only = 50
	start = 3000
	vocabulary_size = len(VocabProcessor.vocabulary_)
	indices = np.random.randint(vocabulary_size, size=(plot_only))

	#low_dim_embs = tsne.fit_transform(embedding[start:start+plot_only,:])
	low_dim_embs = tsne.fit_transform(embedding[indices])

	for i in xrange(plot_only):

		x, y = low_dim_embs[i,0], low_dim_embs[i, 1]
		plt.scatter(x, y)
		pos = indices[i]
		pos = np.array([[pos]])
		temp = VocabProcessor.reverse(pos)
		#plt.annotate(vocabulary[i+start], xy=(x, y), xytext=(x,y))
		plt.annotate(temp.next(), xy=(x, y), xytext=(x,y))

	plt.show()



def main():

	start = time.time()
	corpus = 'big_data.txt'
	skip_window = 5

	#build the vocabulary
	VocabProcessor, max_doc_length = build_vocabulary(corpus, skip_window)
	vocabulary_size = len(VocabProcessor.vocabulary_)
	print('Time taken : %f Minutes\n' %((time.time()-start)/60))
	start = time.time()


	#set all flags
	flags = set_flags(vocabulary_size, skip_window, corpus)

	#get all hyper-parameters of the model
	embedd_size = flags.FLAGS.embedd_size
	neg_samples = flags.FLAGS.neg_samples
	eta = flags.FLAGS.eta
	batch_size = flags.FLAGS.batch_size

	#open the datafile
	datafile = open(corpus)
	text = datafile.read().replace('\n', ' ')
	iterator = ngrams(text.split(), flags.FLAGS.n)
	print('n-gram generator created')
	print('completed time ' + str(datetime.now().hour) + ':' + str(datetime.now().minute))


	graph = tf.Graph()
	with graph.as_default():

		trainX = tf.placeholder(tf.int64, shape = [batch_size])
		trainY = tf.placeholder(tf.int64, shape = [batch_size, 1])

		#set up the model parameters
		W_1 = tf.Variable(tf.random_uniform([vocabulary_size, embedd_size], -1.0, 1.0))

		W_2 = tf.Variable(tf.truncated_normal([vocabulary_size, embedd_size], stddev=1.0/math.sqrt(embedd_size)))
		b_2 = tf.Variable(tf.zeros([vocabulary_size]))

		embed = tf.nn.embedding_lookup(W_1, trainX)

		# Compute the average NCE loss for the batch.
		loss = tf.reduce_mean(tf.nn.nce_loss(W_2, b_2, embed, trainY, neg_samples, vocabulary_size))

		# Construct the SGD optimizer using a learning rate of alpha.
		optimizer = tf.train.GradientDescentOptimizer(eta).minimize(loss)

		# Add variable initializer.
		init = tf.initialize_all_variables()

	print('Graph structure building completed')
	print('completed time ' + str(datetime.now().hour) + ':' + str(datetime.now().minute))

	with tf.Session(graph=graph) as session:

		#initialize all variables
		session.run(init)
		print('Graph initialized\n')

		avg_loss = 0

		for step in xrange(flags.FLAGS.max_steps):
			#print('Iteration : %d' %(step+1))

			iterator, batchX, batchY = generate_batch(iterator, flags, VocabProcessor)
			feed_dict = {trainX:batchX, trainY:batchY}

			_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
			avg_loss += loss_val

			if step%500==0 and step>0:
				avg_loss /= 500
				# The average loss is an estimate of the loss over the last 2000 batches.
				print("Average loss at Step %d is : %f" %(step, avg_loss))
				avg_loss = 0
				print('completed time ' + str(datetime.now().hour) + ':' + str(datetime.now().minute))


		word2Vec = session.run(W_1+W_2)

	print('Embedding created')
	print('Embedding learning time: %f Minutes\n' %((time.time()-start)/60))
	np.save('word2Vec', word2Vec)


	#visualize the embedding
	plot_embedding(word2Vec, VocabProcessor)
	print('completed time ' + str(datetime.now().hour) + ':' + str(datetime.now().minute))

	#bonus
	index = VocabProcessor.transform(['service'])
	index = np.array(list(index)).flatten()
	print('Index of the word SERVICE is %d\n' %(index[0]))
	print('Maximum Document Length %d\n' %max_doc_length)


if __name__ == '__main__':
	main()


