import random
import os
import glob
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import generateLanguage

"""
Generates numpy data that can be input into the model one epoch at a time
+ trials are the list of possible sets of trials that can be shown to the model
+ num_trials represents the number of trials per epoch
+ size_layer_message is the size of the message layer in the neural network
+ size_layer_phonology is the size of the phonological layer in the network
NOTE: Our raw data size is given by num_trials*(length of the word phonology for a given trial)
"""
def generateModel_data(trials, num_trials,
					   verbose = False):
	# Start assembling big 'batches' of data from a number of trials
	for _ in range(num_trials):
		trial = random.choice(trials) # Sampling with replacement
		# Creates the input and output data numpy arrays
		if _ == 0:
			input_data = trial.get_message_np()
			output_data = trial.get_phonology_np()
		# And keeps adding to them until we have added num_trials number of trials
		else:
			input_data = np.vstack((input_data, trial.get_message_np()))
			output_data = np.vstack((output_data, trial.get_phonology_np()))
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
def generateModel_epoch(num_epochs, trials, num_trials, num_batches, num_timeSteps, 
						verbose = False):
	for i in range(num_epochs):
		if verbose:
			print("\tStarting epoch: {}".format(i))
		input_data, output_data = generateModel_data(trials = trials,
													 num_trials = num_trials,
													 verbose = verbose)
		yield generateModel_batch(input_data = input_data,
								 output_data = output_data, 
								 num_batches = num_batches, 
								 num_timeSteps = num_timeSteps,
								 verbose = verbose)

def generateModel(modelInfo,
				  verbose = False):
	graph = tf.Graph()
	nodes = {}
	random.seed(modelInfo['modelSeed'])
	with graph.as_default():
		# Define nodes
		with tf.name_scope('Message'):
			#x = tf.placeholder(tf.int32, [modelInfo['num_batches'], modelInfo['num_timeSteps']], name = 'Message') # Input
			#x = tf.placeholder(tf.int32, [None, modelInfo['num_timeSteps']], name = 'Message') # Input
			x = tf.placeholder(tf.int32, [None, None], name = 'Message') # Input
			batch_size = tf.shape(x)[0]
			modelInfo['input'] = x.name
			nodes['x'] = x
			rnn_inputs = tf.one_hot(x, modelInfo['size_layer_message'], name = 'Message_RNNFeed') # Transformation into what the RNN layers can understand
		with tf.name_scope('RNN_Hidden'):
			cell = tf.contrib.rnn.BasicRNNCell(modelInfo['size_layer_hidden'], name = 'RNN_Cells') # RNN
			#init_state = tf.zeros([modelInfo['num_batches'], modelInfo['size_layer_hidden']], name = 'RNN_Initial') # Initial state of RNN
			max_length = tf.placeholder(tf.int32) # Max length of the RNN unfolding
			init_state = tf.zeros([batch_size, modelInfo['size_layer_hidden']], name = 'RNN_Initial') # Initial state of RNN
			rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state = init_state, sequence_length = max_length)
			modelInfo['hidden_state_init'] = init_state.name
			modelInfo['hidden_state_curr'] = final_state.name
			modelInfo['max_length_name'] = max_length.name
			nodes['init_state'] = init_state
			nodes['final_state'] = final_state
			nodes['max_length'] = max_length
		with tf.name_scope('Phonology'):
			with tf.variable_scope('Softmax'): # Calculated output
				W = tf.get_variable('W', [modelInfo['size_layer_hidden'], modelInfo['size_layer_phonology']])
				b = tf.get_variable('b', [modelInfo['size_layer_phonology']], initializer=tf.constant_initializer(0.0))
			#y = tf.placeholder(tf.int32, [modelInfo['num_batches'], modelInfo['num_timeSteps']], name = 'Phonology') # What the output should be
			#y = tf.placeholder(tf.int32, [None, modelInfo['num_timeSteps']], name = 'Phonology') # What the output should be
			y = tf.placeholder(tf.int32, [None, None], name = 'Phonology') # What the output should be
			nodes['y'] = y
		# Define edges
		with tf.name_scope('Outputs'):
			#logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs, [-1, modelInfo['size_layer_hidden']]), W) + b, [modelInfo['num_batches'], modelInfo['num_timeSteps'], modelInfo['size_layer_phonology']], name = 'Logits')
			#logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs, [-1, modelInfo['size_layer_hidden']]), W) + b, [batch_size, modelInfo['num_timeSteps'], modelInfo['size_layer_phonology']], name = 'Logits')
			logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs, [-1, modelInfo['size_layer_hidden']]), W) + b, [batch_size, max_length, modelInfo['size_layer_phonology']], name = 'Logits')
			predictions = tf.nn.softmax(logits, name = 'Predictions')
			modelInfo['predictions'] = predictions.name
		with tf.name_scope('Errors'):
			losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name = 'Losses')
			total_loss = tf.reduce_mean(losses, name = 'Total_loss')
			train_step = tf.train.AdagradOptimizer(modelInfo['learning_rate']).minimize(total_loss)
			nodes['losses'] = losses
			nodes['total_loss'] = total_loss
			nodes['train_step'] = train_step
	return graph, nodes, modelInfo

def saveLosses(modelName, training_losses, saveLossesLocation,
			   header = ['modelName', 'epoch', 'loss']):
	for i, loss in enumerate(training_losses):
		generateLanguage.recordResponse(fileName = saveLossesLocation, response = {'modelName':modelName, 'epoch':i, 'loss':loss}, header = header) 

"""
Actually train the model
"""
def trainNetwork(modelInfo, graph, nodes, trials_training_singles,
				 saveGraphLocation = None,
				 saveVariablesLocation = None,
				 saveEveryNEpochs = 5,
				 saveLossesLocation = None,
				 verbose = True):
	tf.reset_default_graph()
	if verbose:
		if 'modelName' in modelInfo.keys():
			print("Beginning to train model {}".format(modelInfo['modelName']))
	with tf.Session(graph = graph) as sess:
		sess.run(tf.global_variables_initializer())
		training_losses = []
		if saveGraphLocation != None:
			# Draw model #
			writer = tf.summary.FileWriter(saveGraphLocation)
			writer.add_graph(sess.graph)
			# To view model type: tensorboard --logdir==training:/Users/StevenSchwering/Documents/Psychology/Labs/LCNL/Research/Current/WordChoice_Model/ModelGraphs --host=127.0.0.1
		if saveVariablesLocation != None:
			saved_epochs = range(saveEveryNEpochs, modelInfo['num_epochs']-saveEveryNEpochs+1, saveEveryNEpochs)
			saver = tf.train.Saver(max_to_keep = len(saved_epochs) + 2)
		random.seed(modelInfo['trainingSeed']) # Doesn't actually do anything
		for i, epoch in enumerate(generateModel_epoch(num_epochs = modelInfo['num_epochs'], 
													  trials = trials_training_singles, 
													  num_trials = modelInfo['num_trials'],
													  num_batches = modelInfo['num_batches'], 
													  num_timeSteps = modelInfo['num_timeSteps'],
													  verbose = verbose)):
			for step, (X, Y) in enumerate(epoch):
				training_loss = 0
				training_state = np.zeros((modelInfo['num_batches'], modelInfo['size_layer_hidden']))
				tr_losses, training_loss_, training_state, _ = sess.run([nodes['losses'], nodes['total_loss'], nodes['final_state'], nodes['train_step']], feed_dict={nodes['x']:X, nodes['y']:Y, nodes['init_state']:training_state, nodes['max_length']:modelInfo['num_timeSteps']})
				training_loss += training_loss_
			training_losses.append(training_loss/100)
			training_loss = 0
			if verbose:
				print("\t\tEpoch: {} -- Loss: {}".format(i, training_losses[-1]))
			if saveVariablesLocation != None:
				if i == 0:
					save_path = saver.save(sess, (saveVariablesLocation + 'ModelEpoch_'), global_step = i)
					if verbose:
						print("\t\tModel saved in path: %s" % save_path)
				elif i in saved_epochs:
					save_path = saver.save(sess, (saveVariablesLocation + 'ModelEpoch_'), global_step = i)
					if verbose:
						print("\t\tModel saved in path: %s" % save_path)
				elif i == modelInfo['num_epochs'] - 1:
					save_path = saver.save(sess, (saveVariablesLocation + 'ModelEpoch_'), global_step = i)
					if verbose:
						print("\t\tModel saved in path: %s" % save_path)
	if saveLossesLocation != None:
		saveLosses(modelInfo['modelName'], training_losses, saveLossesLocation)
	return training_losses

if __name__ == '__main__':
	verbose = True
	save = False
	langInfo = {'seed' : 896575376869}
	pathName = os.getcwd() + '/Languages/Lang_' + str(langInfo['seed'])
	langDirTrials = pathName + '/Trials.csv'
	langDirInfo = pathName + '/Info.csv'
	if not os.path.isfile(langDirTrials):
		if not os.path.exists(pathName):
			os.makedirs(pathName)
		langInfo.update({'numTargets' : 10, # Number of messages that CAN have multiple words associated with them
						 'numPhon' : 18, # Number of phonemes in the language
						 'lenPhon_Total' : [3], # Possible word lengths
						 'numAmbigTargets' : 10, # Number of targets that are 'ambiguous' -- the number of messages that DO have multiple words associated with them
						 'numInterfering' : 10, # Number of target messages that have interfering words associated with them
						 'lenPhon_Interfering' : 2, # Length of the phonological interference})
						 'testReserve' : 0.10})
		random.seed(langInfo['seed'])
		# Start by generating the stimuli used for the model
		if verbose:
			print("Random seed: {}".format(langInfo['seed']))
		words, messages_target = generateLanguage.generateTargetMessages(numTargets = langInfo['numTargets'], 
																		numPhon = langInfo['numPhon'],
														   				lenPhon_Total = langInfo['lenPhon_Total'],
												   		   				numAmbigTargets = langInfo['numAmbigTargets'],
												   		   				verbose = verbose)
		print(messages_target)
		words, messages_primes = generateLanguage.generatePrimes(messages_target = messages_target,
						 								 		numInterfering = langInfo['numInterfering'],
						 								 		lenPhon_Total = langInfo['lenPhon_Total'],
						 								 		numPhon = langInfo['numPhon'],
						 								 		lenPhon_Interfering = langInfo['lenPhon_Interfering'],
						 								 		words = words,
						 								 		verbose = verbose)
		print(messages_primes)
		trials_training_singles = generateLanguage.generateTrials(messages_target = messages_target,
										 						 messages_primes = messages_primes,
										 						 lenPhon_Interfering = langInfo['lenPhon_Interfering'],
										 						 testReserve = langInfo['testReserve'],
										 						 verbose = verbose)
		if save:
			generateLanguage.saveLanguage(singles = trials_training_singles, langDirTrials = langDirTrials, langDirInfo = langDirInfo, langInfo = langInfo)
	else:
		trials_training_singles, langInfo = generateLanguage.regenerateFromFile(langDirTrials = langDirTrials, langDirInfo = langDirInfo, langInfo = langInfo, verbose = verbose)	# Model parameters
	modelInfo = {'modelSeed':9021093912,
				 'trainingSeed':129308,
				 'seed':langInfo['seed'], # Seed of the language on which the model is trained
				 'learning_rate':0.005, # Low learning rate makes advancements slow but more stable
				 'num_trials':10, # Number of trials per epoch
				 'num_epochs':1, # Number of samples of trials
				 'num_timeSteps':3, # Number of time steps over which time is calculated
				 'num_batches':1, # Number of batches into which we divide the data
				 'size_layer_message':len(trials_training_singles), # Size of the x layer -- input into the model determined by number of 'messages' in language
				 'size_layer_phonology':langInfo['numPhon'], # Size of y layer -- number of phonological units in language
				 'size_layer_hidden':50} # Size of hidden layer -- should really be a hypoerparameter
	# Generate the nodes and edges
	random.seed(langInfo['seed'])
	graph, nodes, modelInfo = generateModel(modelInfo = modelInfo,
				  		  		 			verbose = verbose)
	# Saving data
	if save:
		modelInfo['modelName'] = '238789562'
		saveGraphLocation = os.getcwd() + "/Models/Model" + str(modelInfo['modelName'])
		saveVariablesLocation = os.getcwd() + "/Models/Model" + str(modelInfo['modelName']) + "/"
		saveLossesLocation = os.getcwd() + "/Models/Model" + str(modelInfo['modelName']) + "/ModelLosses.csv"
	else:
		modelName = 'TEST'
		saveGraphLocation = None
		saveVariablesLocation = None
		saveLossesLocation = None
	# RUN MODEL #
	training_losses = trainNetwork(modelInfo = modelInfo,
									graph = graph,
									nodes = nodes,
									trials_training_singles = trials_training_singles,
				  					saveGraphLocation = saveGraphLocation,
				  					saveVariablesLocation = saveVariablesLocation,
				  					saveLossesLocation = saveLossesLocation,
				  					verbose = verbose)
	plt.plot(training_losses)
	plt.show()