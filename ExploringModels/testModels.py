import tensorflow as tf
import os
import numpy as np
from generateLanguage import *
from generateModels import *

num_batches = 1
size_layer_hidden = 50
np.set_printoptions(precision = 2, suppress = True)


def testModel(trials_training_singles,
			  num_trials = 50,
			  num_epochs = 1, 
			  num_batches = 5, 
			  num_timeSteps = 10, 
			  size_layer_hidden = 50):
	tf.reset_default_graph()
	x = tf.placeholder(tf.int32, [num_batches, num_timeSteps], name = 'Message')
	training_state = np.zeros((num_batches, size_layer_hidden))
	y = tf.placeholder(tf.int32, [num_batches, num_timeSteps], name = 'Phonology')
	graph = os.getcwd() + '/Models/Model238789562/ModelEpoch_-99.meta'
	#print(tf.get_default_graph().get_tensor_by_name('Predictions:0'))
	with tf.Session() as sess:
		restoredGraph = tf.train.import_meta_graph(graph)
		restoredGraph.restore(sess, tf.train.latest_checkpoint(checkpoint_dir = os.getcwd() + '/Models/Model238789562'))
		"""operations = (tf.get_default_graph().get_operations())
		for op in operations:
			print(op.values())"""
		for i, epoch in enumerate(generateModel_epoch(num_epochs = num_epochs, 
													  trials = trials_training_singles, 
													  numTrials = num_trials,
													  num_batches = num_batches, 
													  num_timeSteps = num_timeSteps)):
			training_state = np.zeros((num_batches, size_layer_hidden))
			for step, (X, Y) in enumerate(epoch):
				print("Input: \n{}".format(X))
				print("Output: \n{}".format(Y))
				predictions, training_state = sess.run(['Outputs/Predictions:0', 'RNN_Hidden/rnn/while/Exit_3:0'], feed_dict={'Message/Message:0':X, 'RNN_Hidden/RNN_Initial:0':training_state})
				print(predictions)

if __name__ == '__main__':
	langInfo = {'seed' : 896575376869}
	pathName = os.getcwd() + '/Languages/Lang_' + str(langInfo['seed'])
	langDirTrials = pathName + '/Trials.csv'
	langDirInfo = pathName + '/Info.csv'
	trials_training_singles, langInfo = regenerateFromFile(langDirTrials = langDirTrials, langDirInfo = langDirInfo, langInfo = langInfo)
	testModel(trials_training_singles)