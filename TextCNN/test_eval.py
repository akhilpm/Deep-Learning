import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from gensim.models import word2vec
from cNN_model import TextCNN
import helpers

def main():
	# Eval Parameters
	tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
	#tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
	tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
	tf.flags.DEFINE_integer('embedd_size', 150, 'dimension of the embedding vector space')
	tf.flags.DEFINE_integer('no_classes', 4, 'Number of classes')

	# Misc Parameters
	tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
	tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
	tf.flags.DEFINE_integer("sequence_length", 32, "max length of an email")
	tf.flags.DEFINE_string("model_path", "runs/model-1600", "path to model file")
	tf.flags.DEFINE_string("datafile", "data/test_data.csv", "path to test data file")
	FLAGS = tf.flags.FLAGS


	#load the data
	#testX, testY = helpers.load_test_data()
	''' Note: testY should be in a vector form, not binary representation of the label''' 
	testX = helpers.load_test_data(FLAGS.datafile)
	total_comments = len(testX)
	print('Total no. of comments : %d' %total_comments)	
	testY = None

	#restore vocabulary
	embedding = word2vec.Word2Vec.load('data/word2vec_gensim')

	#checkpoint_file = tf.train.latest_checkpoint('/home/akhil/Documents/job/sentiment/runs/1477980153/checkpoints')
	#checkpoint_file = '/home/akhil/Documents/job/sentiment/runs/model-53500'
	checkpoint_file = FLAGS.model_path
	print(checkpoint_file)
	graph = tf.Graph()
	with graph.as_default():
		conf = tf.ConfigProto(
			allow_soft_placement = FLAGS.allow_soft_placement,
			log_device_placement = FLAGS.log_device_placement)

		sess = tf.Session(config=conf)
		with sess.as_default():
			#saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			#saver = tf.train.import_meta_graph('/home/akhil/Documents/job/sentiment/runs/model-53500.meta')
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)

			dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
			print(type(dropout_keep_prob))
			y = graph.get_operation_by_name('label').outputs[0]
			embedd_sentence = graph.get_operation_by_name('embedding/embedd_sentence').outputs[0]
			predictions = graph.get_operation_by_name('accuracy/predictions').outputs[0]
			#pscore = graph.get_operation_by_name('output/probs').outputs[0]

			batches = helpers.batch_iter_testdata(testX, FLAGS.batch_size)
			all_predictions = np.empty(shape=[0, FLAGS.no_classes])

			for batch in batches:
				embedd_batch = helpers.embedding_lookup(embedding, batch, FLAGS.batch_size, 
				FLAGS.sequence_length, FLAGS.embedd_size)
				feed_dict = {
				embedd_sentence : embedd_batch,
				dropout_keep_prob : 1.0
				}
				batch_precitions = sess.run(predictions, feed_dict)
				all_predictions = np.append(all_predictions, batch_precitions, axis=0)
				#all_probs = np.concatenate(all_probs, batch_probs)

	if testY is not None:
		match = 1.0*(all_predictions==testY)
		match = np.mean(match, axis=1)
		match = 1.0*(match==1)
		print('Total no. of test samples : %d' %len(testY))
		print('Accuracy: {:g}'.format(np.mean(match)))

	if testY is None:
		for prediction in all_predictions:
			print(prediction)


if __name__ == '__main__':
	main()

