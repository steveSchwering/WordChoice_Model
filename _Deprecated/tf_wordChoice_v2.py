import random
import os
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from generateLanguage import *

"""
Generates numpy data that can be input into the model one epoch at a time
+ trials are the list of possible sets of trials that can be shown to the model
+ numTrials represents the number of trials per epoch
+ size_layer_message is the size of the message layer in the neural network
+ size_layer_phonology is the size of the phonological layer in the network
NOTE: Our raw data size is given by numTrials*(length of the word phonology for a given trial)
"""
def generateModel_data(trials, numTrials,
					   verbose = False):
	if verbose:
		print("Generating trials for model")
	# Start assembling big 'batches' of data from a number of trials
	for _ in range(numTrials):
		trial = random.choice(trials) # Sampling with replacement
		# Creates the input and output data numpy arrays
		if _ == 0:
			input_data = trial.get_message_np()
			output_data = trial.get_phonology_np()
		# And keeps adding to them until we have added numTrials number of trials
		else:
			input_data = np.vstack((input_data, trial.get_message_np()))
			output_data = np.vstack((output_data, trial.get_phonology_np()))
	if verbose:
		print("\tGenerated {} inputs".format(len(input_data)))
		print("\tGenerated {} outputs".format(len(output_data)))
	input_data = input_data.flatten()
	output_data = output_data.flatten()
	return input_data, output_data

"""
Takes data from generateModel_data and breaks it into minibatches
			***CODE LARGELY ADAPTED FROM r2rt.com***
+ input_data is input data from previous function
+ output_data is output data from previous function
+ num_batches is the number of batches into which we will divide the work
+ num_timeSteps is the number of timesteps back in time that the error is propagated
NOTE: Realized later that a lot of this can be implemented via the tf.one_hot command
"""
def generateModel_batch(input_data, output_data, num_batches, num_timeSteps,
						verbose = False):
	# First we partition the data into batches and stack them
	#- Remember that each batch will be trained on a parallel model sharing the weights
    assert len(input_data) == len(output_data)
    data_length = len(input_data)
    batch_partition_length = data_length // num_batches
    data_x = np.zeros([num_batches, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([num_batches, batch_partition_length], dtype=np.int32)
    for i in range(num_batches):
        data_x[i] = input_data[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = output_data[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_timeSteps for truncated backprop
    epoch_size = batch_partition_length // num_timeSteps
    for i in range(epoch_size):
        x = data_x[:, i * num_timeSteps:(i + 1) * num_timeSteps]
        y = data_y[:, i * num_timeSteps:(i + 1) * num_timeSteps]
        yield (x, y)

"""
Generate data and have it incrementally pass it to model
"""
def generateModel_epoch(num_epochs, trials, numTrials, num_batches, num_timeSteps, size_layer_x, size_layer_y, verbose):
	for i in range(num_epochs):
		input_data, output_data = generateModel_data(trials = trials,
													 numTrials = numTrials,
													 verbose = verbose)
		yield generateModel_batch(input_data = input_data,
								 output_data = output_data, 
								 num_batches = num_batches, 
								 num_timeSteps = num_timeSteps,
								 verbose = verbose)

"""
Actually train the model
"""
def train_network(trials_training_singles, num_trials, num_batches, num_timeSteps, num_epochs, size_layer_message, size_layer_phonology, size_layer_hidden, saveLocation,
				  drawModel = True,
				  verbose = True,
				  verbose_trunc = True):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		training_losses = []
		if drawModel:
			# Draw model #
			writer = tf.summary.FileWriter(saveLocation + '/Models')
			writer.add_graph(sess.graph)
			# To view model type: tensorboard --logdir==training:/Users/StevenSchwering/Documents/Psychology/Labs/LCNL/Research/Current/WordChoice_Model/ModelGraphs --host=127.0.0.1
		for i, epoch in enumerate(generateModel_epoch(num_epochs = num_epochs, 
													  trials = trials_training_singles, 
													  numTrials = num_trials,
													  num_batches = num_batches, 
													  num_timeSteps = num_timeSteps, 
													  size_layer_x = size_layer_message, 
													  size_layer_y = size_layer_phonology, 
													  verbose = verbose)):
			if verbose:
				print("|||||||||||||||||||||\nStarting epoch: {}\n|||||||||||||||||||||".format(i))
			for step, (X, Y) in enumerate(epoch):
				training_loss = 0
				training_state = np.zeros((num_batches, size_layer_hidden))
				tr_losses, training_loss_, training_state, _ = sess.run([losses, total_loss, final_state, train_step], feed_dict={x:X, y:Y, init_state:training_state})
				training_loss += training_loss_
				if verbose:
					print("\t\tCurrent loss at step {} : {}".format(step + 1, (training_loss/100)))
			training_losses.append(training_loss/100)
			training_loss = 0
			if not verbose and verbose_trunc:
				print("Epoch: {} -- Loss: {}".format(i, training_losses[-1]))
	return training_losses

def generateModel(learning_rate, num_trials, num_epochs, num_timeSteps, num_batches, size_layer_message, size_layer_phonology, size_layer_hidden, saveLocation,
				  verbose = False):
	pass

if __name__ == '__main__':
	seed = 896575376869
	random.seed(seed)
	verbose = False
	# Start by generating the stimuli used for the model
	if verbose:
		print("Random seed: {}".format(seed))
	numTargets = 10 # Ge10nerally, keep this number below 500, as may start causing difficulties finding words w/ random
	numPhon = 18
	lenPhon_Total = [3]
	numAmbig = numTargets
	words, messages_target = generateTargetMessages(numTargets = numTargets, 
													numPhon = numPhon,
									   				lenPhon_Total = lenPhon_Total,
							   		   				numAmbig = numAmbig,
							   		   				verbose = verbose)
	numInterfering = int(numTargets / 1)
	lenPhon_Interfering = 2
	words, messages_primes = generatePrimes(messages_target = messages_target,
	 								 		numInterfering = numInterfering,
	 								 		lenPhon_Total = lenPhon_Total,
	 								 		numPhon = numPhon,
	 								 		lenPhon_Interfering = lenPhon_Interfering,
	 								 		words = words,
	 								 		verbose = verbose)
	testReserve = 0.10
	trials_training_singles, trials_training_pairs, trials_testing = generateTrials(messages_target = messages_target,
					 								 								messages_primes = messages_primes,
					 								 								lenPhon_Interfering = lenPhon_Interfering,
					 								 								testReserve = testReserve,
					 								 								verbose = verbose)
	# Model parameters
	learning_rate = 0.005
	num_trials = len(trials_training_singles)*10 # Number of trials per epoch
	num_epochs = 1000 # Number of groups of trials
	num_timeSteps = 3 # Number of time steps over which time is calculated
	num_batches = 2 # Number of batches into which we divide our data
	# Model structure
	size_layer_message = numTargets
	size_layer_phonology = numPhon
	size_layer_hidden = 200
	# Saving data
	saveLocation = os.getcwd() + "/SavedModels"
		# Define model nodes
	with tf.name_scope('Message'):
		x = tf.placeholder(tf.int32, [num_batches, num_timeSteps], name = 'Message') # Input
		rnn_inputs = tf.one_hot(x, size_layer_message, name = 'Message_RNNFeed') # Transformation into what the RNN layers can understand
	with tf.name_scope('RNN_Hidden'):
		cell = tf.contrib.rnn.BasicRNNCell(size_layer_hidden, name = 'RNN_Cells') # RNN
		init_state = tf.zeros([num_batches, size_layer_hidden], name = 'RNN_Initial') # Initial state of RNN
		rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
	with tf.name_scope('Phonology'):
		with tf.variable_scope('Softmax'): # Calculated output
			W = tf.get_variable('W', [size_layer_hidden, size_layer_phonology])
			b = tf.get_variable('b', [size_layer_phonology], initializer=tf.constant_initializer(0.0))
		y = tf.placeholder(tf.int32, [num_batches, num_timeSteps], name = 'Phonology') # What the output should be
	# Define edges
	with tf.name_scope('Outputs'):
		logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs, [-1, size_layer_hidden]), W) + b, [num_batches, num_timeSteps, size_layer_phonology], name = 'Logits')
		predictions = tf.nn.softmax(logits, name = 'Predictions')
	with tf.name_scope('Errors'):
		losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name = 'Losses')
		total_loss = tf.reduce_mean(losses, name = 'Total_loss')
		train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)
	# RUN MODEL #
	training_losses = train_network(trials_training_singles = trials_training_singles,
				  					num_trials = num_trials,
				  					num_batches = num_batches,
				  					num_timeSteps = num_timeSteps,
				  					num_epochs = num_epochs,
				  					size_layer_message = size_layer_message,
				  					size_layer_phonology = size_layer_phonology,
				  					size_layer_hidden = size_layer_hidden,
				  					saveLocation = saveLocation,
				  					verbose = verbose)
	plt.plot(training_losses)
	plt.show()