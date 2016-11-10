import time
import numpy as np
import tensorflow as tf
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datetime import datetime
from tflearn.data_utils import VocabularyProcessor


def plot_embedding(embedding, VocabProcessor):
	""" plots the produced embedding using TSNE """

	#words = ['becase', 'becauase', 'becaue', 'becaus', 'because', 'becausee', 'becauz', 'becaz', 'becoge', 'becos', 'becose', 'becous',
	# 'becouse', 'becox', 'becoz', 'becoze', 'becsuse', 'becuase', 'becuse', 'becuz', 'becz', 'bcz', 'bcze', 'bczi', 'bcuse', 'bcus', 'bcse',
	#  'bcs', 'bcozz', 'bcoz', 'bcos', 'bco', 'bcaz', 'bcause', 'bcaus', 'bcas']
	words = ['service', 'good', 'not', 'of', 'my', 'very', 'in', 'store', 'your', 'was']

	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500000)
	indices = []
	print('Total no. of words = %d\n' %len(words))

	for i in xrange(len(words)):
		#index = vocabulary.index(words[i])
		index = VocabProcessor.transform([words[i]])
		index = np.array(list(index)).flatten()[0]
		indices.append(index)

	vocabulary_size = len(VocabProcessor.vocabulary_)
	indices = np.array(indices, dtype=np.int)
	#print(indices)

	#low_dim_embs = tsne.fit_transform(embedding[start:start+plot_only,:])
	low_dim_embs = tsne.fit_transform(embedding[indices])
	#low_dim_embs = tsne.fit_transform(embedding)

	for i in xrange(len(words)):

		x, y = low_dim_embs[i,0], low_dim_embs[i, 1]
		plt.scatter(x, y)
		#plt.annotate(vocabulary[i+start], xy=(x, y), xytext=(x,y))
		plt.annotate(words[i], xy=(x, y), xytext=(x,y))

	plt.show()


def plot_analogy(embedding, VocabProcessor):
	""" plots the produced embedding using TSNE """

	#word_pairs = [{'good':'awesome'}, {'bad':'worst'}]
	word_pairs = [{'happy':'satisfied'}, {'unhappy':'dissatisfied'}]
	#word_pairs = [{'problem':'resolved'}, {'doubt':'cleared'}] 
	ax = plt.axes() 

	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500000)
	indices = []

	for i in xrange(len(word_pairs)):
		#index1, index2 = vocabulary.index(word_pairs[i].keys()[0]), vocabulary.index(word_pairs[i].values()[0])
		index = VocabProcessor.transform([word_pairs[i].keys()[0]])
		index1 = np.array(list(index)).flatten()[0]
		#print index1
		index = VocabProcessor.transform([word_pairs[i].values()[0]])
		index2 = np.array(list(index)).flatten()[0]
		indices.extend((index1, index2))

	vocabulary_size = len(VocabProcessor.vocabulary_)
	indices = np.array(indices, dtype=np.int)
	print(indices)

	#low_dim_embs = tsne.fit_transform(embedding[start:start+plot_only,:])
	low_dim_embs = tsne.fit_transform(embedding[indices])
	print low_dim_embs

	for i in xrange(len(word_pairs)):

		x1, y1 = low_dim_embs[2*i,0], low_dim_embs[2*i, 1]
		x2, y2 = low_dim_embs[2*i+1,0], low_dim_embs[2*i+1, 1]

		print(x1, y1, x2, y2)

		plt.scatter(x1, y1)
		plt.scatter(x2, y2)
		#if i==0:
		plt.arrow(x1, y1, x2-x1, y2-y1, head_width=0.3, head_length=0.6, fc='k', ec='k')
		#plt.arrow(x1, y1, x1-x2, y1-y2, head_width=0.3, head_length=0.6, fc='k', ec='k')		
		#plt.annotate(vocabulary[i+start], xy=(x, y), xytext=(x,y))
		if i%2==0:
			plt.annotate(word_pairs[i].keys()[0], xy=(x1, y1), xytext=(x1,y1), weight='bold')
			plt.annotate(word_pairs[i].values()[0], xy=(x2, y2), xytext=(x2,y2), weight='bold')
		else:
			plt.annotate(word_pairs[i].values()[0], xy=(x1, y1), xytext=(x1,y1), weight='bold')
			plt.annotate(word_pairs[i].keys()[0], xy=(x2, y2), xytext=(x2,y2), weight='bold')				

	plt.show()


def test_plot():

	x1 = 1
	y1 = 1
	x2 = 10
	y2 = 10

	plt.scatter(x1, y1)
	plt.scatter(x2, y2)
	plt.arrow(x1, y1, x2, y2, head_width=0.3, head_length=0.4, fc='k', ec='k')
	plt.show()


def main():
	#f = open('vocabulary.txt', 'r')
	#vocabulary = pickle.load(f)
	max_document_length = 51
	VocabProcessor = VocabularyProcessor(max_document_length)
	VocabProcessor = VocabProcessor.restore('vocabulary.txt')

	embedding = np.load('word2Vec.npy')
	print('Data loading completed @ ' + str(datetime.now().hour) + ':' + str(datetime.now().minute))

	plot_embedding(embedding, VocabProcessor)
	#plot_analogy(embedding, VocabProcessor)

	#test_plot()


if __name__ == '__main__':
	main()
