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
#from imblearn.over_sampling import SMOTE


class GetConfig(object):
	"""Small config."""
	learning_rate = 0.1
	max_grad_norm = 5
	num_hid_layers = 1
	input_size = 26
	hid_size1 = 20
	hid_size2 = 10
	output_size = 2
	max_epoch = 10
	keep_prob = 0.5
	l2_reg = 1.9
	batch_size = 10000
	checkpoint_every = 900000

def change_label_score(y):
	y[ y==1.0 ] = 0.7
	y[ y==0.0 ] = 0.3
	return y


class NeuralNetwork:

	def __init__(self, config, is_training):

		#all required variables
		'''
		self.W1 = tf.get_variable('w1', [config.input_size, config.hid_size1], dtype=tf.float32)
		self.W2 = tf.get_variable('w2', [config.hid_size1, config.output_size], dtype=tf.float32)
		self.b1 = tf.get_variable('b1', [config.hid_size1], dtype=tf.float32)
		self.b2 = tf.get_variable('b2', [config.output_size], dtype=tf.float32)
		'''
		self.WI = tf.Variable(tf.random_normal([config.input_size, config.hid_size1], stddev=1.7))
		self.W1 = tf.Variable(tf.random_normal([config.hid_size1, config.hid_size2], stddev=0.1))
		self.WO = tf.Variable(tf.random_normal([config.hid_size2, config.output_size], stddev=1.7))
		self.bI = tf.Variable(tf.random_normal([config.hid_size1], stddev=1.7))
		self.b1 = tf.Variable(tf.random_normal([config.hid_size2], stddev=0.1))
		self.bO = tf.Variable(tf.random_normal([config.output_size], stddev=1.7))
		self.is_training = is_training

		self.normWI = tf.norm(self.WI)
		self.normWO = tf.norm(self.WO)

		#placeholders for input and output
		self.X = tf.placeholder(tf.float32, 
			[config.batch_size, config.input_size], name='X')
		self.y = tf.placeholder(tf.float32, 
			[config.batch_size, config.output_size], name='y')

		#class weights to partially solve the class imbalance problem
		#ratio = 9723.0 / 277.0
		#class_weights = tf.constant([ratio, 1 - ratio])
		class_weights = tf.constant([0.50803677,  31.60700685])
		'''
		for balanced weighting based on number of samples in each class compute the weights as
		n_samples / (n_classes * np.bincount(y))
		which in this case is
		1662290.0 / (2*np.array([1658540, 26260]))
		array([   0.50113051,  221.63866667])
		class_weights = tf.constant([0.50113051,  221.63866667])
		'''

		#forward propagation
		self.h1 = tf.nn.sigmoid(tf.matmul(self.X, self.WI) + self.bI)
		self.h1 = tf.nn.dropout(self.h1, config.keep_prob)
		self.h2 = tf.nn.sigmoid(tf.matmul(self.h1, self.W1) + self.b1)
		self.drop_out = tf.nn.dropout(self.h2, config.keep_prob)
		self.logits = tf.matmul(self.drop_out, self.WO) + self.bO
		#self.logits = tf.nn.sigmoid(tf.matmul(self.drop_out, self.WO) + self.bO)
		self.weighted_logits = tf.multiply(self.logits, class_weights)
		self.predictions = tf.argmax(self.logits, 1, name='predictions')

		#compute loss
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			labels=self.y, logits=self.weighted_logits))
		self.loss += config.l2_reg*(tf.nn.l2_loss(self.WI) + \
			tf.nn.l2_loss(self.WO)) + tf.nn.l2_loss(self.bI)  + \
			tf.nn.l2_loss(self.bO)
		#self.loss += config.l2_reg*(tf.nn.l2_loss(self.WI) + tf.nn.l2_loss(self.W1) + \
		#	tf.nn.l2_loss(self.WO)) + tf.nn.l2_loss(self.bI) + tf.nn.l2_loss(self.b1) + \
		#	tf.nn.l2_loss(self.bO)

		#compute accuracy
		correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))

		#return is testing. No need to define optimizer
		if not is_training:
			return

		#tools for back-propagation
		self.optimizer = tf.train.AdamOptimizer(1e-3)
		#self.optimizer = tf.train.GradientDescentOptimizer(0.001)
		self.train_op = self.optimizer.minimize(self.loss)


def main():
	print('started @ ' + str(datetime.now().hour) + ':' + str(datetime.now().minute))
	config = GetConfig()
	#scaler = preprocessing.StandardScaler().fit(testX)
	#testX = scaler.transform(testX)
	scaler = joblib.load('scaler.pkl')

	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():
			nnet = NeuralNetwork(config, is_training=True)

			global_step = tf.Variable(0, trainable=True, name='global_step')
			timestamp = str(int(time.time()))

			saver = tf.train.Saver(tf.global_variables())
			out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
			print('Writing to {}\n'.format(out_dir))

			#checkpoint directory
			checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
			checkpoint_prefix = os.path.join(checkpoint_dir, 'modelNN')
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)

			#initialize all varibales
			sess.run(tf.global_variables_initializer())
			print(sess.run(nnet.normWI))
			print(sess.run(nnet.normWO))
			current_step = 0

			for i in range(config.max_epoch):
				train_data = pd.read_csv('train_data.csv', chunksize=config.batch_size)
				batch_count = 0
				print('EPOCH NO: %d' %(i+1))
				for batch in train_data:
					#current_step = tf.train.global_step(sess, global_step)
					current_step += config.batch_size
					print('processing batch no: %d at step: %d' %(batch_count+1, current_step))
					data = batch.values.astype(np.float32)
					trainX, trainY = data[:,:-2], data[:, -2:]
					trainX = trainX[:, 1:]
					#trainX = scaler.transform(trainX)
					#trainY = change_label_score(trainY)

					#reshape trainX to [batch_size, num_steps, ip_size]
					try:
						trainX = trainX.reshape(config.batch_size, config.input_size)
					except:
						print('ignoring the last batch')
						break
					feed_dict = {nnet.X:trainX, nnet.y:trainY}
					_, step = sess.run([nnet.train_op, global_step], feed_dict)
					#print('step count : %d' %(current_step))

					time_str = datetime.now().isoformat()
					if current_step % config.checkpoint_every == 0:
						path = saver.save(sess, checkpoint_prefix, global_step=current_step)
						print('Saved model checkpoint to {}\n'.format(path))
					batch_count += 1
					if batch_count > 30:
						break



			print('training completed @ ' + str(datetime.now().hour) + ':' + str(datetime.now().minute))
			#do testing on the test data
			print(sess.run(nnet.normWI))
			print(sess.run(nnet.normWO))
			nnet.is_training = False
			test_data = pd.read_csv('test_data.csv',chunksize=config.batch_size)
			test_accuracy = 0.0
			test_loss = 0.0
			test_predictions = []
			test_correct = []
			count = 0
			for batch in test_data:
				data = batch.values.astype(np.float32)
				testX, testY = data[:,:-2], data[:, -2:]
				testX = testX[:, 1:]
				zero_sum, one_sum = np.sum(testY[:,0]), np.sum(testY[:,1])
				zero_sum = zero_sum*1.0
				print('zeros : %d, ones : %d, ratio : %f' %(zero_sum, one_sum, np.float(zero_sum / one_sum)))
				#testX = scaler.transform(testX)
				#testY = change_label_score(testY)
				try:
					testX = testX.reshape(config.batch_size, config.input_size)
				except:
					print('ignoring the last batch')
					break

				feed_dict = {nnet.X:testX, nnet.y:testY}
				ta, tl, pr, logits = sess.run([nnet.accuracy, nnet.loss,
				 nnet.predictions, nnet.logits], feed_dict)
				print('batch %d : accuracy: %f' %(count+1, ta))
				#print(logits[:10])
				test_loss += tl * config.batch_size
				test_accuracy += ta * config.batch_size
				test_predictions = np.append(test_predictions, pr)
				test_correct = np.append(test_correct, np.argmax(testY, axis=1))
				count += 1
				#if count > 100:
				#	break

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
