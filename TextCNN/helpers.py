import numpy as np
import re
import pandas as pd
import random


imp_category_map = {'category0':0,
'category1':1,
'category2':2,
'category3':3
}

# A string to clean sentences
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
	string = re.sub(r"\(", " ", string)
	string = re.sub(r"\)", " ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	string = re.sub(r"\"", "", string)
	return string.strip().lower()


# A utility to load data
def load_data(no_of_classes, datafile):

	emails = pd.read_csv(datafile, delimiter=',', encoding='utf-7')
	classes = np.empty(shape=[0, no_of_classes])
	data = []
	count = 0
	for idx, email in emails.iterrows():
		labels = email['cat'].lower()
		labels = labels.split('|')
		#print labels
		for label in labels:
			if label in imp_category_map.keys():
				if count==0:
					temp = np.zeros((1, no_of_classes))
					temp[0, imp_category_map[label]] = 1 
					classes = np.append(classes, temp, axis=0)
					count += 1
					mail_body = email['body']
					mail_body = clean_str(mail_body)
					data.append(mail_body)
				else:
					classes[-1, imp_category_map[label]] = 1
					count += 1
		count = 0			

	data = np.array(data)
	return [data, classes]


def load_test_data(datafile):
	emails = pd.read_csv(datafile, encoding='utf-7')
	data = []
	for idx, email in emails.iterrows():
		mail_body = clean_str(email['body'])
		data.append(mail_body)
	data = np.array(data)
	return data



def embedding_lookup(embedding, sentences, batch_size, sequence_length, embedd_size):
	no_of_datapoints = min(len(sentences), batch_size)
	embedd_sentence = np.zeros((no_of_datapoints, sequence_length, embedd_size))
	print(embedd_sentence.shape)
	for i in range(len(sentences)):
		sent = sentences[i].split()
		sent = [clean_str(word) for word in sent]
		no_of_words = min(sequence_length, len(sent))
		for j in range(no_of_words):
			if sent[j] in embedding.wv:
				embedd_sentence[i, j] = embedding.wv[sent[j]]
	return embedd_sentence



#get a batch of data
def batch_iter(x_text, y_text, batch_size, num_epochs, shuffle=True):
	""" generates a batch iterator """
	data = zip(x_text, y_text)
	data_size = len(data)
	num_batches_per_epoch = int(data_size/batch_size) + 1
	#print(num_batches_per_epoch)

	for epoch in xrange(num_epochs):
		#print('hello world')
		#shuffle the data
		if shuffle:
			random.shuffle(data)
		shuffled_data = data

		for batch_num in xrange(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num+1)*batch_size, data_size)
			print start_index, end_index
			yield shuffled_data[start_index:end_index]


def batch_iter_testdata(data, batch_size):
	data_size = len(data)
	num_batches = int(data_size/batch_size) + 1

	for batch_num in xrange(num_batches):
		start_index = batch_num * batch_size
		end_index = min((batch_num+1)*batch_size, data_size)
		yield data[start_index:end_index]
