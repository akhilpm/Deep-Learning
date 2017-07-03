import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from datetime import datetime


def lstm_cell(size):
	return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

def attn_cell(config):
	return  tf.contrib.rnn.DropoutWrapper(lstm_cell(config.hid_size),
	output_keep_prob=config.keep_prob)

def get_batch(config):
	seq = np.zeros([config.batch_size, config.num_steps, config.input_size], dtype=np.float32)
	tgt = np.zeros([config.batch_size, 1], dtype=np.float32)

def change_label_score(y):
	y[ y==1.0 ] = 0.7
	y[ y==0.0 ] = 0.3
	return y

class GetConfig(object):
	"""Small config."""
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_steps = 1
	hid_size = 50
	max_epoch = 4
	keep_prob = 0.5
	l2_reg = 100.9
	batch_size = 10000
	input_size = 26
	checkpoint_every = 5000


class LSTM:

	def __init__(self, config, is_training):

		self.batch_size = config.batch_size
		self.num_steps = config.num_steps
		self.is_training = is_training

		self.sequence = tf.placeholder(tf.float32, 
			[config.batch_size, config.num_steps, config.input_size], name='ip_sequence')
		self.target = tf.placeholder(tf.float32, 
			[config.batch_size, 2], name='target')
		#L2 loss : to be used in summary generation
		l2_loss = tf.constant(0.0)

		#computation of a multi-layer RNN
		cell = tf.contrib.rnn.MultiRNNCell(
			[attn_cell(config) for _ in range(config.num_layers)], state_is_tuple=True)
		self.initial_state = cell.zero_state(config.batch_size, tf.float32)

		#get the output of RNN by looking 'num_steps' backward
		state = self.initial_state
		with tf.variable_scope('RNN'):
			for time_step in range(config.num_steps):
				if time_step>0:
					tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(self.sequence[:, time_step, :], state)
		self.final_state = state

		#reshape the output to make it compatible with final softmax layer
		output = tf.reshape(cell_output, [-1, config.hid_size])

		#class weights to partially solve the class imbalance problem
		ratio = 9723.0 / 277.0
		class_weights = tf.constant([ratio, 1 - ratio])


		#softmax layer
		with tf.name_scope('output'):
			softmax_w = tf.get_variable('softmax_w', [config.hid_size, 2], dtype=tf.float32)
			softmax_b = tf.get_variable('softmax_b', [2], dtype=tf.float32)
			self.logits = tf.sigmoid(tf.matmul(output, softmax_w) + softmax_b)
			self.predictions = tf.argmax(self.logits, 1, name='predictions')

			l2_loss += tf.nn.l2_loss(softmax_w)
			l2_loss += tf.nn.l2_loss(softmax_b)
			self.weighted_logits = tf.multiply(self.logits, class_weights)

		#compute mean cross entropy loss	
		with tf.name_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.target, 
				logits=self.logits)
			self.loss = tf.reduce_sum(losses) / self.batch_size + config.l2_reg*l2_loss
	
		#compute the accuracy
		with tf.name_scope('accuracy'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.target, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

		#return is testing. No need to define optimizer
		if not self.is_training:
			return

		self.optimizer = tf.train.AdamOptimizer(1e-3)
		self.train_op = self.optimizer.minimize(self.loss)


		


def main():
	print('started @ ' + str(datetime.now().hour) + ':' + str(datetime.now().minute))
	config = GetConfig()
	#scaler = preprocessing.StandardScaler().fit(testX)
	#testX = scaler.transform(testX)
	scaler = joblib.load('scaler/scaler.pkl')

	#build the graph and start the computation
	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():
			#initialize the LSTM object
			lstm = LSTM(config, is_training=True)

			global_step = tf.Variable(0, trainable=False, name='global_step')
			timestamp = str(int(time.time()))

			saver = tf.train.Saver(tf.global_variables())
			out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
			print('Writing to {}\n'.format(out_dir))

			#checkpoint directory
			checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
			checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)

			#initialize all varibales
			sess.run(tf.global_variables_initializer())

			for i in range(config.max_epoch):
				train_data = pd.read_csv('train_data.csv', chunksize=config.batch_size)
				batch_count = 0
				print('EPOCH NO: %d' %(i+1))
				for batch in train_data:
					current_step = tf.train.global_step(sess, global_step)
					print('processing batch no: %d at step: %d' %(batch_count+1, current_step))
					data = batch.values.astype(np.float32)
					trainX, trainY = data[:,:-2], data[:, -2:]
					trainX = scaler.transform(trainX)
					trainY = change_label_score(trainY)

					#reshape trainX to [batch_size, num_steps, ip_size]
					try:
						trainX = trainX.reshape(config.batch_size, 1, config.input_size)
					except:
						print('ignoring the last batch')
						break
					feed_dict = {lstm.sequence:trainX, lstm.target:trainY}
					_, step = sess.run([lstm.train_op, global_step], feed_dict)
					print('step count : %d' %(step))

					time_str = datetime.now().isoformat()
					if current_step % config.checkpoint_every == 0:
						path = saver.save(sess, checkpoint_prefix, global_step=current_step)
						print('Saved model checkpoint to {}\n'.format(path))
					batch_count += 1
					#if batch_count==10:
					#	break

			print('training completed @ ' + str(datetime.now().hour) + ':' + str(datetime.now().minute))
			#do testing on the test data
			lstm.is_training = False
			test_data = pd.read_csv('test_data.csv',chunksize=config.batch_size)
			test_accuracy = 0.0
			test_loss = 0.0
			test_predictions = []
			test_correct = []
			count = 0
			for batch in test_data:
				data = batch.values.astype(np.float32)
				testX, testY = data[:,:-2], data[:, -2:]
				zero_sum, one_sum = np.sum(testY[:,0]), np.sum(testY[:,1])
				zero_sum = zero_sum*1.0
				print('zeros : %d, ones : %d, ratio : %f' %(zero_sum, one_sum, np.float(zero_sum / one_sum)))
				testX = scaler.transform(testX)
				testY = change_label_score(testY)
				try:
					testX = testX.reshape(config.batch_size, 1, config.input_size)
				except:
					print('ignoring the last batch')
					break

				feed_dict = {lstm.sequence:testX, lstm.target:testY}
				ta, tl, pr, logits = sess.run([lstm.accuracy, lstm.loss,
				 lstm.predictions, lstm.logits], feed_dict)
				print('batch %d : accuracy: %f' %(count+1, ta))
				print logits[:10]
				test_loss += tl * config.batch_size
				test_accuracy += ta * config.batch_size
				test_predictions = np.append(test_predictions, pr)
				test_correct = np.append(test_correct, np.argmax(testY, axis=1))
				count += 1

			print('testing completed @ ' + str(datetime.now().hour) + ':' + str(datetime.now().minute))	
			test_accuracy = test_accuracy / (count * config.batch_size)
			test_loss = test_loss / (count * config.batch_size)
			print('test accuracy : {:g}' .format(test_accuracy))
			print('test loss : {:g}' .format(test_loss))
			test_predictions = np.array(test_predictions).flatten()
			test_correct = np.array(test_correct, dtype=np.float32).flatten()

			print(confusion_matrix(test_correct, test_predictions))
			print('precision : %f ' %precision_score(test_correct, test_predictions))
			print('recall : %f ' %recall_score(test_correct, test_predictions))
			print('finished @ ' + str(datetime.now().hour) + ':' + str(datetime.now().minute))

			

if __name__ == '__main__':
	main()