import numpy as np
import tensorflow as tf
import time
from datetime import datetime
from tensorflow.contrib import learn
import os
import re
from cNN_model import TextCNN
import helpers
import gensim
from gensim.models import word2vec
import tflearn

def make_separate_labels(Y):
	no_classes = Y.shape[1]
	newY = np.zeros([no_classes, Y.shape[0], 2])
	for i in range(no_classes):
		temp1 = Y[:,i].reshape(len(Y), 1)
		newY[i] = tflearn.data_utils.to_categorical(temp1, 2)
	return newY



#=========== DEFINE ALL PARAMETERS ================

#Model Hyperparameters
tf.flags.DEFINE_integer('embedd_size', 150, 'dimension of the embedding vector space')
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer('num_filters', 100, 'no. of filters per  filter size')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
tf.flags.DEFINE_float('l2_reg', 0.01, 'regularization parameter for the weights')

#Training parameters
tf.flags.DEFINE_integer('batch_size', 64, 'Batch size while training')
tf.flags.DEFINE_integer('num_epochs', 20, 'no. of iterations to train')
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 500)")

# Misc Parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow softplacement to CPU when GPU is not available')
tf.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of operations on devices')
tf.flags.DEFINE_integer('no_classes', 4, 'Number of classes')
tf.flags.DEFINE_string('train_data', 'data/train_data.csv', 'train data file')

FLAGS = tf.flags.FLAGS

""" Load the data shuffle it and split into test and train sets """
#x_text, y_text = helpers.load_data()
x_text, y_text = helpers.load_data(FLAGS.no_classes, FLAGS.train_data)
print(y_text.shape)
max_document_length = np.int(np.mean([len(x.split(" ")) for x in x_text]))
print('max document length : %d' %max_document_length)
embedding = word2vec.Word2Vec.load('data/word2vec_gensim')
#VocabProcessor = VocabularyProcessor(max_document_length)
#VocabProcessor = VocabProcessor.restore('vocabulary.txt')

#X = np.array(list(VocabProcessor.transform(x_text)))
#print('X shape : %d %d' %(X.shape))

seed = np.random.randint(100)
rand = np.random.RandomState(seed)
total_comments = len(x_text)
print('Total no. of comments : %d' %total_comments)
indices = rand.randint(total_comments, size=total_comments)
#indices = indices.reshape(total_comments, 1)
x_shuffled = x_text[indices]
y_shuffled = y_text[indices]

x_train, x_dev = x_shuffled[-100:], x_shuffled[:-100]
y_train, y_dev = y_shuffled[-100:], y_shuffled[:-100]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
devy = make_separate_labels(y_dev)
embedd_devx = helpers.embedding_lookup(embedding, x_dev, FLAGS.batch_size, 
	max_document_length, FLAGS.embedd_size)


""" Training process """
with tf.Graph().as_default():
	conf = tf.ConfigProto(
		allow_soft_placement = FLAGS.allow_soft_placement,
		log_device_placement = FLAGS.log_device_placement)

	sess = tf.Session(config=conf)
	with sess.as_default():
		cnn = TextCNN(sequence_length=max_document_length, num_classes=FLAGS.no_classes,
			embedd_size=FLAGS.embedd_size, batch_size=FLAGS.batch_size,
			filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))), num_filters=FLAGS.num_filters,
			l2_reg=FLAGS.l2_reg)

		#global_step = tf.Variable(0, trainable=False, name='global_step')
		optimizer = tf.train.AdamOptimizer(1e-3)
		grads_and_vars = optimizer.compute_gradients(cnn.loss)
		train_op = optimizer.apply_gradients(grads_and_vars)		


		#Output directory for models and summaries
		timestamp = str(int(time.time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
		print('Writing to {}\n'.format(out_dir))

		#Summaries of Loss and Accuracy
		loss_summary = tf.summary.scalar('Loss', cnn.loss)
		acc_summary = tf.summary.scalar('Accuracy', cnn.accuracy)

		#Train summaries
		train_summary_op = tf.summary.merge([loss_summary, acc_summary])
		train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
		train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

		#Dev summaries
		dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
		dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
		dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

		#checkpoint directory
		checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
		checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		saver = tf.train.Saver(tf.global_variables())

		#initialize all varibales
		sess.run(tf.global_variables_initializer())


		def train_step(embedd_sentence, y_batch, y):
			""" one step while training """

			feed_dict = {
			#cnn.trainX : x_batch, 
			cnn.trainY : y_batch,
			cnn.Y : y,
			cnn.embedd_sentence : embedd_sentence,
			cnn.dropout_keep_prob : FLAGS.dropout_keep_prob,
			}

			_, summaries, loss, accuracy = sess.run([train_op, train_summary_op, 
				cnn.loss, cnn.accuracy], feed_dict)

			time_str = datetime.now().isoformat()
			print('{}:, loss {:g}, acc {:g}'.format(time_str, loss, accuracy))
			train_summary_writer.add_summary(summaries)


		def dev_step(embedd_sentence, y_batch, y, writer=None):
			""" Cross validate the model """

			feed_dict = {
			#cnn.trainX :x_batch,
			cnn.trainY : y_batch,
			cnn.Y : y,
			cnn.embedd_sentence : embedd_sentence,
			cnn.dropout_keep_prob : 1.0
			}

			summaries, loss, accuracy = sess.run([dev_summary_op,
				cnn.loss, cnn.accuracy], feed_dict)

			time_str = datetime.now().isoformat()
			print('{}:, loss {:g}, acc {:g}'.format(time_str, loss, accuracy))

			if writer:
				writer.add_summary(summaries)

		#generate the batches
		batches = helpers.batch_iter(x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)


		#for each batch do the following
		current_step = 0
		for batch in batches:
			x_batch, y_batch = zip(*batch)
			y_batch = np.array(y_batch)
			#print y_batch

			#do one training step
			embedd_sentence = helpers.embedding_lookup(embedding, x_batch, FLAGS.batch_size, 
				max_document_length, FLAGS.embedd_size)
			y = make_separate_labels(y_batch)
			train_step(embedd_sentence, y_batch, y)
			current_step += FLAGS.batch_size

			if current_step % FLAGS.evaluate_every == 0:
				print('\nEvaluation')
				dev_step(embedd_devx, y_dev, devy, writer=dev_summary_writer)
				print('')

			if current_step % FLAGS.checkpoint_every == 0:
				path = saver.save(sess, checkpoint_prefix, global_step=current_step)
				print('Saved model checkpoint to {}\n'.format(path))

	feed_dict = {
		cnn.trainY : y_dev,
		cnn.Y : devy,
		cnn.embedd_sentence : embedd_devx,
		cnn.dropout_keep_prob : 1.0
		}
	test_result = sess.run([cnn.prediction], feed_dict)
	y_dev = y_dev.astype(np.int32)
	for i in range(y_dev.shape[0]):
		print y_dev[i], test_result[0][i]



