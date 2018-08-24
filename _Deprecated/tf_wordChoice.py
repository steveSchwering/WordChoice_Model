import random
import os
import glob
import pandas
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
Notes:
-- Feedback helps to overcome noisy inputs
-- -- Train NN on comprehension and production trials. Goal of comprehension trial is to get accurate prediction of upcoming phonological input. Goal of the production trial is to get accurate production of phonological information.
-- -- This input and output would be captured by the same layer. Comprehension could be imagined as feeding up into this 'phonological' layer whereas production could be imagined as feeding down into a motor articulation layer.
"""

"""
What question does this model answer?
+ We are trying to model competition between words chosen to label an object
+ In this model, we are interested in how the effect of phonological interference
+ of produced words described in Koranda & MacDonald (in press) data can be
+ described as interference between phonological components of words. In the Koranda & MacDonald
+ task, participants are asked to name a picture that has 'low name agreement'.
+ Potential labels for this image are near synonyms like 'jacket' and 'coat'.
+ Preceding this image by an unambiguous image that only has one name, like
+ 'jackal', may affect the word chosen if the name is phonologically similar,
+ but not identical, to one of the choices. The effect of this interference
+ can be predicted. The cited work has shown that the interfering word reduces
+ the probability that the phonologically similar word will be producing,
+ suggesting that prior productions impact subsequent productions. Yet, why do
+ phonologically similar words interfere? What is the mechanism by which phonologically
+ similar words affect each other? To understand the mechanisms driving these observations
+ I am building a neural network that can be trained to produce labels for images.

Model philosophy
+ To capture the characteristics described above, the model employs several common
+ machine learning principles. Largely, this model is inspired by word comprehension
+ models like the TRACE model and word comprehension/production models like in the
+ verb ending model of Seidenberg & Joanisse (1998). The following are the model
+ principles that I find important:
+
+ The model must be a simple recurrent network
+ 	It must process information over time so that we can see how interference between
+ 	competing words develops over that time span.
+ The model must 'produce' words using a phonological code from a 'message'
+ 	This is important to create interference from similar phonologies

Model structure
+ Message layer
+ 	Represents the images -- the 'semantic' representation of the words
+ Phonological layer

Training structure
+ The model will be trained to produce individual words and then pairs of words

Testing structure
+ First, the model will 'see' an image and be asked to produce a word associated with that image
+ 	This first word will always be unambiguous. That is, it will only have 1 associated word
+ 	t1 => Activate node associated with image in message layer
+ 	t2, t3, t4 => Produce a word in the phonological output layer
+ Next, the model will 'see' a second image and be asked to produce a word associated with that image
+ 	This second word will always be ambiguous. That is, the word will have at least 2 associated words
+ 	t4 => Activate node associated with image in message layer
+ 	t5, 56, 57 => Produce word in phonological output layer
"""

"""
TestTrial class defines the information being fed to our neural network
+ image<#> indicates the accurate <first/second> message
+ label<#> indicates the accurate <first/second> word
+ phonemes<#> indicates the accurate <first/second> phonemes
"""
class Trial():
	def __init__(self, condition, image1, label1, phonemes1, image2, label2, phonemes2):
		self.condition = condition
		self.image1 = image1
		self.label1 = label1
		self.phonemes1 = phonemes1
		self.image2 = image2
		self.label2 = label2
		self.phonemes2 = phonemes2

	"""
	Outputs one-hot vectors of model input
	"""
	def get_message_np(self, num_messages):
		zeros = np.array([0]*num_messages, dtype = np.float32)
		if self.image2 == None:
			out = np.eye(num_messages)[self.image1].astype(np.float32)
			for _ in range(len(self.phonemes1) - 1):
				out = np.vstack((out, zeros))
			return out
		else:
			out1 = np.eye(num_messages)[self.image1].astype(np.float32)
			out2 = np.eye(num_messages)[[self.image2]].astype(np.float32)
			for _ in range(len(self.phonemes1) - 1):
				out1 = np.vstack((out1, zeros))
			for _ in range(len(self.phonemes2) - 1):
				out2 = np.vstack((out2, zeros))
			return out1, out2
		
	"""
	Outputs one-hot vectors of model output
	"""
	def get_phonology_np(self, num_phonological_units):
		phonemes1 = [int(_) for _ in self.phonemes1]
		if self.image2 == None:
			return np.eye(num_phonological_units)[phonemes1].astype(np.int32)
		else:
			phonemes2 = [int(_) for _ in self.phonemes2]
			return np.eye(num_phonological_units)[phonemes1].astype(np.int32), np.eye(num_phonological_units)[phonemes2].astype(np.int32)

"""
This function generates message:word pairings in a dictionary format
+ numTargets tells us how many 'messages' there are
+ 	Remember a single message can have multiple words words associated with it (e.g. 'jacket' or 'coat')
+ numPhon tells us how many phonological units there are
+ lenPhon_Total tells us how many phonological units are associated with each word
+ numAmbig tells us how many of the messages have multiple possible words associated with it
+ 	We want some words to be ambiguous and some to not be ambiguous, so this is usually set to half of numTargets
"""
def generateTargetMessages(numTargets, numPhon, lenPhon_Total, numAmbig,
						   verbose = False):
	assert numAmbig <= numTargets
	# Which messages are going to have to have two words associated with it?
	shared = ([1] * int(numAmbig)) + ([0] * (int(numTargets) - int(numAmbig)))
	random.shuffle(shared)
	# Storage of words and messages
	words = []
	messages_target= {}
	# Generate each message and associated possible words
	if verbose:
		print("Generating targets:")
	# message is the index of the image
	# isShared is whether the message has more than 1 associated word
	for message, isShared in zip(range(numTargets), shared):
		if isShared == 1:
			while True:
				word1 = []
				word2 = []
				# Randomly adds phonemes together to create a word. Phonemes can repeat.
				for phoneme in range(random.choice(lenPhon_Total)):
					word1.append(random.choice(range(numPhon)))
				for phoneme in range(random.choice(lenPhon_Total)):
					word2.append(random.choice(range(numPhon)))
				# Checks if the two words with a message share any phonemes (they should not)
				# Also checks if the words already exist (they should not)
				if (not bool(set(word1) & set(word2))) & ((word1 not in words) & (word2 not in words)):
					words.append(word1)
					words.append(word2)
					messages_target[message] = random.choice([[word1, word2], [word2, word1]])
					break
			if verbose:
				print("\t{} -- {}".format(message, messages_target[message]))
		elif isShared == 0:
			while True:
				word = []
				for phoneme in range(random.choice(lenPhon_Total)):
					word.append(random.choice(range(numPhon)))
				# Just checks if the word created is already a word
				if word not in words:
					words.append(word)
					messages_target[message] = [word]
					break
			if verbose:
				print("\t{} -- {}".format(message, messages_target[message]))
	return words, messages_target

"""
This function generates the primes that will be presented before the targets
+ messages_target are the messages-phonological pairs already created by the generateTargetMessages function
+ numInterfering indicates the number of messages_target pairs that will have interfering message primes
+ 	Default should be all of them
+ lenPhon_Total indicates possible word lengths of the language
+ lenPhon_Interfering indicates how may of the elements of the phonological list starting from index 0 are overlapping between prime and target
+ words keeps track of what words were already created. Passed in from prior function
"""
def generatePrimes(messages_target, numInterfering, lenPhon_Total, lenPhon_Interfering, words,
				   verbose = False):
	assert numInterfering <= len(messages_target)
	# Which messages are going to have interfering words presented before them?
	interfering = ([1] * int(numInterfering)) + ([0] * (len(messages_target) - int(numInterfering)))
	random.shuffle(interfering)
	messages_primes = {}
	if verbose:
		print("Generating primes:")
	# Generate the priming words with a few restrictions
	for primeNum, isInterfering in zip(messages_target.keys(), interfering):
		if isInterfering:
			while True:
				messages_primes[primeNum] = []
				# Generates an interfering word for each word associated with a message
				for word in messages_target[primeNum]:
					word = word[:lenPhon_Interfering]
					wordLen = random.choice(lenPhon_Total)
					# Makes sure the new prime word is as long as or longer than the target
					if wordLen - lenPhon_Interfering < 1:
						continue 
					# Generates new word by append to a truncated target word
					for phoneme in range(wordLen - lenPhon_Interfering):
						word.append(random.choice(range(numPhon)))
					# Checks if the word already exists AND makes sure it doesn't overlap with ANY phonological component of ANY other word in the target set
					if (word not in words) & (not bool(set(word[-(wordLen - lenPhon_Interfering):]) & set(messages_target[primeNum][0]))) & (not bool(set(word[-(wordLen - lenPhon_Interfering):]) & set(messages_target[primeNum][1]))):
						messages_primes[primeNum].append(word)
				# Checks to make sure each target word has an overlapping prime; appends words to tracker
				if len(messages_target[primeNum]) == len(messages_primes[primeNum]):
					for word in messages_primes[primeNum]:
						words.append(word)
					break
				# Else restart
				else:
					continue
			if verbose:
				print("\t{}. {} -- {}".format(primeNum, messages_target[primeNum], messages_primes[primeNum]))
		else:
			while True:
				messages_primes[primeNum] = []
				for word in messages_target[primeNum]:
					word = []
					for phoneme in range(random.choice(lenPhon_Total)):
						word.append(random.choice(range(numPhon)))
					# Checks if the word already exists AND makes sure it doesn't overlap with ANY target word
					if (not bool(set(word) & set(messages_target[primeNum][0]))) & (not bool(set(word) & set(messages_target[primeNum][1]))) & (word not in words):
						messages_primes[primeNum].append(word)
				if len(messages_target[primeNum]) == len(messages_primes[primeNum]):
					for word in messages_primes[primeNum]:
						words.append(word)
					break
				else:
					continue
			if verbose:
				print("\t{}. {} -- {}".format(primeNum, messages_target[primeNum], messages_primes[primeNum]))
	return words, messages_primes

"""
This function generates input and output. The key part here is that messages_primes contains unambiguous words.
That is, the 'message' of each word is connected to only one phonological pattern. These will all act as the primes.
The values of messages_primes are stored by the same key as their corresponding ambiguous 'message' in messages_target.
"""
def generateTrials(messages_target, messages_primes, lenPhon_Interfering,
				   testReserve = .10,
				   verbose = False):
	assert testReserve < 1.00
	# Separate all of the ambiguous primes
	if verbose:
		print("Setting up ambiguous singles")
	trials_ambiguous = []
	for message in messages_target:
		for phonological_code in messages_target[message]:
			phonological_code = [str(_) for _ in phonological_code]
			label = 'artificial_{}'.format('-'.join(phonological_code))
			trials_ambiguous.append(Trial(condition= None, image1 = message, label1 = label, phonemes1 = phonological_code, image2 = None, label2 = None, phonemes2 = None))
	# Separate all of the unambiguous primes
	if verbose:
		print("Setting up unambiguous singles")
	messageNum = len(messages_target.keys()) - 1
	trials_unambiguous = []
	for message in messages_primes:
		for phonological_code in messages_primes[message]:
			messageNum += 1
			phonological_code = [str(_) for _ in phonological_code]
			label = 'artificial_{}'.format('-'.join(phonological_code))
			trials_unambiguous.append(Trial(condition= None, image1 = messageNum, label1 = label, phonemes1 = phonological_code, image2 = None, label2 = None, phonemes2 = None))
	# Generate all possible pairs of primes and targets
	trials_interfering = []
	trials_noninterfering = []
	if verbose:
		print("Setting up primes")
	for message_target in messages_target:
		for message_prime in messages_primes:
			for m_t in messages_target[message_target]:
				m_t = [str(_) for _ in m_t]
				label2 = 'artificial_{}'.format('-'.join(m_t))
				for m_p in messages_primes[message_prime]:
					m_p = [str(_) for _ in m_p]
					label1 = 'artificial_{}'.format('-'.join(m_p))
					if m_t[:lenPhon_Interfering] == m_p[:lenPhon_Interfering]:
						trials_interfering.append(Trial(condition = 'interfering', image1 = message_prime, label1 = label1, phonemes1 = m_p, image2 = message_target, label2 = label2, phonemes2 = m_t))
					else:
						trials_noninterfering.append(Trial(condition = 'noninterfering', image1 = message_prime, label1 = label1, phonemes1 = m_p, image2 = message_target, label2 = label2, phonemes2 = m_t))
	random.shuffle(trials_interfering)
	random.shuffle(trials_noninterfering)
	# Divide them into testing and training sets
	# - For interfering items
	num_interfere_train = int(len(trials_interfering)*(1-testReserve))
	trials_interfering_train = trials_interfering[:num_interfere_train]
	trials_interfering_test = trials_interfering[num_interfere_train:]
	# - For noninterfering items
	num_noninterfere_train = int(len(trials_noninterfering)*(1-testReserve))
	trials_noninterfering_train = trials_noninterfering[:num_noninterfere_train]
	trials_noninterfering_test = trials_noninterfering[num_noninterfere_train:]
	# Define training and testing trials
	trials_training_singles = trials_ambiguous + trials_unambiguous
	trials_training_pairs = trials_noninterfering_train + trials_noninterfering_train
	if verbose:
		print("\tDefined training set of singletons: {} trials".format(len(trials_training_singles)))
		print("\tDefined training set of pairs: {} trials".format(len(trials_training_pairs)))
	trials_testing = trials_noninterfering_test + trials_interfering_test
	if verbose:
		print("\tDefined testing set of interfering and noninterfering pairs: {} trials".format(len(trials_testing)))
	return trials_training_singles, trials_training_pairs, trials_testing

"""
Generates numpy data that can be input into the model one epoch at a time
+ trials are the list of possible sets of trials that can be shown to the model
+ numTrials represents the number of trials per epoch
+ size_layer_message is the size of the message layer in the neural network
+ size_layer_phonology is the size of the phonological layer in the network
NOTE: Our raw data size is given by numTrials*(length of the word phonology for a given trial)
"""
def generateModel_data(trials, numTrials, size_layer_message, size_layer_phonology,
					   verbose = False):
	if verbose:
		print("Generating trials for model")
	# Start assembling big 'batches' of data from a number of trials
	for _ in range(numTrials):
		trial = random.choice(trials) # Sampling with replacement
		# Creates the input and output data numpy arrays
		if _ == 0:
			input_data = trial.get_message_np(size_layer_message)
			output_data = trial.get_phonology_np(size_layer_phonology)
		# And keeps adding to them until we have added numTrials number of trials
		else:
			input_data = np.vstack((input_data, trial.get_message_np(size_layer_message)))
			output_data = np.vstack((output_data, trial.get_phonology_np(size_layer_phonology)))
	if verbose:
		print("\tGenerated {} inputs".format(len(input_data)))
		print("\tGenerated {} outputs".format(len(output_data)))
	# TODO: Consider extending np.array representation of message through the entire phonological code. The activated 'jackal' from the last production
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
def generateModel_batch(input_data, output_data, num_batches, num_timeSteps, size_layer_x, size_layer_y,
						verbose = False):
	# First we partition the data into batches and stack them
	#- Remember that each batch will be trained on a parallel model sharing the weights
	batch_partition_length = len(input_data) // num_batches
	data_x = np.zeros([num_batches, batch_partition_length, size_layer_x], dtype = np.float32)
	data_y = np.zeros([num_batches, batch_partition_length, size_layer_y], dtype = np.float32)
	if verbose:
		print("Dividing the input into minibatches of length {}".format(batch_partition_length))
		print("\tDimensions of message is {}".format(np.shape(data_x)))
		print("\tDimensions of phonology is {}".format(np.shape(data_y)))
	for i in range(num_batches):
		data_x[i] = input_data[(batch_partition_length*i):(batch_partition_length*(i+1))]
		data_y[i] = output_data[(batch_partition_length*i):(batch_partition_length*(i+1))]
	# Divide the minibatches into chunks based on the number of timesteps over which error is propagated
	chunk_size = batch_partition_length // num_timeSteps
	if verbose:
		print("Dividing each batch into {} minibatches over which error is propagated".format(chunk_size))
	for i in range(chunk_size):
		x = data_x[:, i * num_timeSteps:(i + 1) * num_timeSteps]
		y = data_y[:, i * num_timeSteps:(i + 1) * num_timeSteps]
		if (verbose):
			print("\t{}. Smaller chunk of message is dimensionality {} feeding into phonology of dimensionality {}".format(i+1, np.shape(x), np.shape(y)))
		yield (x, y)

"""
Divide mini-batches
"""
def generateModel_epoch(num_epochs, trials, numTrials, num_batches, num_timeSteps, size_layer_x, size_layer_y, verbose):
	for i in range(num_epochs):
		input_data, output_data = generateModel_data(trials = trials,
													 numTrials = numTrials,
													 size_layer_message = size_layer_x,
													 size_layer_phonology = size_layer_y,
													 verbose = verbose)
		yield generateModel_batch(input_data = input_data,
								 output_data = output_data, 
								 num_batches = num_batches, 
								 num_timeSteps = num_timeSteps, 
								 size_layer_x = size_layer_x, 
								 size_layer_y = size_layer_y,
								 verbose = verbose)

"""
Actually train the model
"""
def train_network(trials_training_singles, num_trials, num_batches, num_timeSteps, num_epochs, size_layer_message, size_layer_phonology, size_layer_hidden, 
				  drawModel = True,
				  verbose = True):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		if drawModel:
			# Draw model #
			writer = tf.summary.FileWriter(os.getcwd() + "/ModelGraphs")
			writer.add_graph(sess.graph)
			# To view model type: tensorboard --logdir==training:/Users/StevenSchwering/Documents/Psychology/Labs/LCNL/Research/Current/WordChoice_Model/ModelGraphs --host=127.0.0.1
		training_losses = []
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
	return training_losses

"""
Creates and combines together RNN cells
"""
def rnn_cell(rnn_input, state):
	with tf.variable_scope('rnn_cell', reuse=True):
		W = tf.get_variable('W', [size_layer_message + size_layer_hidden, size_layer_hidden])
		b = tf.get_variable('b', [size_layer_hidden], initializer=tf.constant_initializer(0.0))
	return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)

if __name__ == '__main__':
	seed = 896575376869
	random.seed(seed)
	verbose = True
	# Start by generating the stimuli used for the model
	if verbose:
		print("Random seed: {}".format(seed))
	numTargets = 200 # Generally, keep this number below 500, as may start causing difficulties finding words w/ random
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
	learning_rate = 0.000001
	num_trials = len(trials_training_singles) # Number of trials per epoch
	num_epochs = 1 # Number of groups of trials
	num_timeSteps = 10 # Number of time steps over which time is calculated
	num_batches = 1 # Number of batches into which we divide our data
	# Model structure
	size_layer_message = numTargets*3
	size_layer_phonology = numPhon
	size_layer_hidden = 20
	# Create the model
	# Create model inputs and outputs
	x = tf.placeholder(tf.float32, [num_batches, num_timeSteps, size_layer_message], name = 'Message')
	y = tf.placeholder(tf.int32, [num_batches, num_timeSteps, size_layer_phonology], name = 'Phonology') 
	init_state = tf.zeros([num_batches, size_layer_hidden], name = 'RNN_initial')
	rnn_inputs = tf.unstack(x, axis = 1, name = 'Message_unstacked')
	"""
	# Create hidden layer and RNN cells
	with tf.variable_scope('rnn_cell'):
		W = tf.get_variable('W', [size_layer_message + size_layer_hidden, size_layer_hidden])
		b = tf.get_variable('b', [size_layer_hidden], initializer = tf.constant_initializer(0.0))
	#- Adding the RNN to the graph
	state = init_state
	rnn_outputs = []
	for rnn_input in rnn_inputs:
		state = rnn_cell(rnn_input, state)
		rnn_outputs.append(state)
	final_state = rnn_outputs[-1]
	"""
	# RNN
	cell = tf.contrib.rnn.BasicRNNCell(size_layer_hidden, name = 'RNN_cell')
	rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state = init_state)
	# Calculate outputs
	"""with tf.variable_scope('softmax'):
		W = tf.get_variable('W', [size_layer_hidden, size_layer_phonology])
		b = tf.get_variable('b', [size_layer_phonology], initializer = tf.constant_initializer(0.0))
	logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
	predictions = [tf.nn.softmax(logit) for logit in logits]"""
	# Turn our y placeholder into a list of labels
	"""y_as_list = tf.unstack(y, num_timeSteps, axis=1)
	# Calculate losses
	losses = [tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit) for logit, label in zip(logits, y_as_list)]
	total_loss = tf.reduce_mean(losses)
	train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)"""
	with tf.variable_scope('Softmax'):
		W = tf.get_variable('W', [size_layer_hidden, size_layer_phonology])
		b = tf.get_variable('b', [size_layer_phonology], initializer=tf.constant_initializer(0.0))
	logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs, [-1, size_layer_hidden]), W) + b, [num_batches, num_timeSteps, size_layer_phonology], name = 'logits')
	predictions = tf.nn.softmax(logits, name = 'Predictions')
	losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits, name = 'Losses')
	total_loss = tf.reduce_mean(losses, name = 'ReduceMean_Losses')
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
				  					verbose = verbose)