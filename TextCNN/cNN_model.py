import tensorflow as tf
import numpy as np

class TextCNN(object):
	""" A  1 layer CNN for text classification. It performs convolution in the first layer, then max-pooling followed by 
		a dropout reqularization """

	def __init__(self, batch_size, sequence_length, num_classes, embedd_size, filter_sizes, num_filters, l2_reg):

		#placeholder variables for input, output & dropout probability
		#self.trainX = tf.placeholder(tf.int32, [None, sequence_length], name='trainX')
		self.trainY = tf.placeholder(tf.int64, [None, num_classes], name='trainY')
		self.Y = tf.placeholder(tf.float32, [num_classes, None, 2], name='label')
		self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

		#L2 loss : to be used in summary generation
		l2_loss = tf.constant(0.0)

		#load the vocabulary
		#f = open('vocabulary.txt', 'r')
		#self.vocabulary = pickle.load(f)

		#load the embedding
		#self.embedding = embedding
		#print('Vocab size : %d' %len(embedding))

		#using CPU since embedding operation is not defined  on GPUs
		with tf.device('/cpu:0'), tf.name_scope('embedding'):			
			#self.embedd_sentence = embedding_lookup(model, sentences, batch_size, sequence_length, embedd_size)
			self.embedd_sentence = tf.placeholder(tf.float32, [None, sequence_length, embedd_size]
				, name='embedd_sentence')
			self.embedd_sentence_expanded = tf.expand_dims(self.embedd_sentence, -1)

		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope("conv-maxpool-%s" %filter_size):
					
				filter_shape = [filter_size, embedd_size, 1, num_filters]
				self.W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W2')
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

				#do the convolution operation
				conv = tf.nn.conv2d(self.embedd_sentence_expanded, self.W2, strides=[1,1,1,1], padding='VALID', name='conv')
				#print(conv.shape)

				#apply activation function for bringing in the non-linearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
				#print(h.shape)

				#apply max-pooling over the convolved outputs
				pooled = tf.nn.max_pool(h, ksize=[1,sequence_length-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID', name='pool')
				pooled_outputs.append(pooled)


		#combine all the pooled features
		total_num_filters = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(pooled_outputs, 3)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_num_filters])

		#add dropout
		with tf.name_scope('dropout'):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)


		#compute unnormalized scores and predictions
		with tf.name_scope('output'):

			#initialize W with xavier filter
			#W = tf.get_variable('W', shape=[total_num_filters, num_classes], initializer=tf.contrib.layers.xavier_initializers())
			Wclass1 = tf.Variable(tf.truncated_normal([total_num_filters, 2], stddev=0.1))
			bclass1 = tf.Variable(tf.constant(0.1, shape=[2]), name='b1')
			Wclass2 = tf.Variable(tf.truncated_normal([total_num_filters, 2], stddev=0.1))
			bclass2 = tf.Variable(tf.constant(0.1, shape=[2]), name='b2')
			Wclass3 = tf.Variable(tf.truncated_normal([total_num_filters, 2], stddev=0.1))
			bclass3 = tf.Variable(tf.constant(0.1, shape=[2]), name='b3')
			Wclass4 = tf.Variable(tf.truncated_normal([total_num_filters, 2], stddev=0.1))
			bclass4 = tf.Variable(tf.constant(0.1, shape=[2]), name='b4')

			l2_loss += tf.nn.l2_loss(Wclass1 + Wclass2 + Wclass3 + Wclass4)
			l2_loss += tf.nn.l2_loss(bclass1 + bclass2 + bclass3 + bclass4)

			#compute scores
			#self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
			self.scores1 = tf.matmul(self.h_drop, Wclass1) + bclass1
			self.probs1 = tf.nn.softmax(self.scores1, name='probs1')
			self.predictions1 = tf.argmax(self.scores1, 1, name='predictions1')
			self.predictions1 = tf.reshape(self.predictions1, [-1, 1])

			self.scores2 = tf.matmul(self.h_drop, Wclass2) + bclass2
			self.probs2 = tf.nn.softmax(self.scores2, name='probs2')
			self.predictions2 = tf.argmax(self.scores2, 1, name='predictions2')
			self.predictions2 = tf.reshape(self.predictions2, [-1, 1])

			self.scores3 = tf.matmul(self.h_drop, Wclass3) + bclass3
			self.probs3 = tf.nn.softmax(self.scores3, name='probs3')
			self.predictions3 = tf.argmax(self.scores3, 1, name='predictions3')
			self.predictions3 = tf.reshape(self.predictions3, [-1, 1])

			self.scores4 = tf.matmul(self.h_drop, Wclass4) + bclass4
			self.probs4 = tf.nn.softmax(self.scores4, name='probs4')
			self.predictions4 = tf.argmax(self.scores4, 1, name='predictions4')
			self.predictions4 = tf.reshape(self.predictions4, [-1, 1])

		#compute mean cross entropy loss
		with tf.name_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y[0], logits=self.scores1) + \
			tf.nn.softmax_cross_entropy_with_logits(labels=self.Y[1], logits=self.scores2) + \
			tf.nn.softmax_cross_entropy_with_logits(labels=self.Y[2], logits=self.scores3) + \
			tf.nn.softmax_cross_entropy_with_logits(labels=self.Y[3], logits=self.scores4)
			self.loss = tf.reduce_mean(losses) + l2_reg*l2_loss

		#compute the accuracy
		with tf.name_scope('accuracy'):
			self.prediction = tf.concat([self.predictions1, self.predictions2, 
				self.predictions3, self.predictions4], 1, name='predictions')
			target = self.trainY
			match = tf.cast(tf.equal(self.prediction, target), tf.float32)
			match = tf.reduce_mean(match, axis=1)
			match = tf.cast(tf.equal(match, 1), tf.float32)
			self.accuracy = tf.reduce_mean(match, name='accuracy')
		#	correct_predictions = tf.equal(self.predictions, tf.argmax(self.trainY, 1))
		#	self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

'''
Y = make_separate_labels(self.trainY)
a = np.array([[0,1,0], [1,1,0], [0,0,1], [1,0,0], [1,0,1]])
b = np.array([[0,1,0], [1,1,1], [1,0,0], [1,0,0], [1,0,1]])
temp = tf.cast(tf.equal(a, b), tf.float32)
match = sess.run(temp)
match = np.mean(match, axis=1)
match = 1.0*(match==1)
accuracy = np.mean(match)
'''