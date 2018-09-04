import glob
import os
import pandas
import sys
import importlib
import tensorflow as tf
import numpy as np
from helperfunctions_WCM import *

"""
Class groups all important elements of testing trials together
+ word_* variables hold information about each word in the test trials
+ sequenceX contains the input sequence into the model. Currently it is in sparse encoding.
+ matchingMessage indicates whether the model is given the same message for the words in the trial
+ matchingPhonology is a list of which kinds phonemes match phoneme-by-phoneme starting left-aligned
"""
class testingTrial():
	def __init__(self, trial1, trial2):
		self.numWords = 2
		# Get information about the first message/word pair
		self.word1_type = trial1['condition']
		self.word1_message = int(trial1['message'])
		self.word1_label = trial1['label']
		self.word1_phonemes = [int(_) for _ in trial1['phonemes'].split('_')]
		# Get information about the second message/word pair
		self.word2_type = trial2['condition']
		self.word2_message = int(trial2['message'])
		self.word2_label = trial2['label']
		self.word2_phonemes = [int(_) for _ in trial2['phonemes'].split('_')]
		# Get input
		zeros1 = np.array([0]*(len(self.word1_phonemes)), dtype = np.float32)
		zeros1[0] = int(self.word1_message)
		zeros1 = zeros1.astype(np.int32)
		zeros2 = np.array([0]*(len(self.word2_phonemes)), dtype = np.float32)
		zeros2[0] = int(self.word2_message)
		zeros2 = zeros2.astype(np.int32)
		sequenceX = np.vstack((zeros1, zeros2))
		self.sequenceX = sequenceX.flatten()
		# Get output
		phonemes1 = np.asarray(self.word1_phonemes).astype(np.int32)
		phonemes2 = np.asarray(self.word2_phonemes).astype(np.int32)
		sequenceY = np.vstack((phonemes1, phonemes2))
		self.sequenceY = sequenceY.flatten()
		# Check if the messages match
		if trial1.message == trial2.message:
			self.matchingMessage = 1
		else:
			self.matchingMessage = 0
		# Check how much of the phonemes match
		self.matchingPhonology = []
		for phon1, phon2 in zip(self.word1_phonemes, self.word2_phonemes):
			if phon1 == phon2:
				self.matchingPhonology.append(1)
			else:
				self.matchingPhonology.append(0)

	def get_input(self):
		zeros1 = np.array([0]*(len(self.word1_phonemes)), dtype = np.int32)
		zeros1[0] = self.word1_message
		zeros2 = np.array([0]*(len(self.word2_phonemes)), dtype = np.int32)
		zeros2[0] = self.word2_message
		return np.concatenate((zeros1,zeros2))

	def get_output(self):
		phonemes1 = np.asarray(self.word1_phonemes).astype(np.int32)
		phonemes2 = np.asarray(self.word2_phonemes).astype(np.int32)
		return np.concatenate((phonemes1,phonemes2))

"""
Generates testing data
"""
def generateTest(trials):
	for trial in trials:
		x = trial.get_input()
		num_timeSteps = len(x)
		x = np.expand_dims(x, axis=0)
		y_hat = trial.get_output()
		y_hat = np.expand_dims(y_hat, axis=0)
		yield(x, y_hat, num_timeSteps)

"""
Run model and see what we get
"""
def getPredictions(modelInfo, graph, nodes, testingTrials, file,
				   verbose = False):
	# gets predictions from what model produces
	graphLoc = modelInfo['modelType'] + "Models/Model" + str(modelInfo['modelName']) + "/"
	if verbose:
		print("Predicting for model {}".format(graphLoc))
	for graph in glob.glob(graphLoc + "*.meta"):
		if verbose:
			print("\tPredicting for graph saved: {}".format(graph))
		data=graph.split('.')[0]
		tf.reset_default_graph()
		x = tf.placeholder(tf.int32,[None, None], name = 'Message')
		y = tf.placeholder(tf.int32, [None, None], name = 'Phonology')
		with tf.Session() as sess:
			restoredGraph = tf.train.import_meta_graph(graph)
			restoredGraph.restore(sess, data)
			for i, test in enumerate(generateTest(trials = testingTrials)):
				training_state = np.zeros((1, modelInfo['size_layer_hidden']))
				(X, Y, num_timeSteps) = test
				if verbose:
					print("Input: \n\t{}".format(X))
					print("Output: \n\t{}".format(Y))
				predictions, training_state = sess.run([modelInfo['predictions'], modelInfo['hidden_state_curr']], feed_dict={modelInfo['input']:X, modelInfo['hidden_state_init']:training_state, modelInfo['max_length_name']:num_timeSteps})
				if verbose:
					print("Predictions: \n\t{}".format(predictions))

"""
Regenerates model from 
"""
def buildModel(modelInfo, testingTrials, file,
			   verbose = False):
	# reconstructs model from tensorflow and returns model
	tf.reset_default_graph()
	try:
		graph, nodes, modelInfo = generateModels.generateModel(modelInfo = modelInfo,
															   verbose = verbose)
	except:
		print("Error recreating model {} of model type {}".format(modelInfo['modelName'], modelType.split('/')[-2]))
		return
	getPredictions(modelInfo = modelInfo,
				   graph = graph,
				   nodes = nodes,
				   testingTrials = testingTrials,
				   file = file,
				   verbose = verbose)

"""
Recreates the language for each model
+ regeneratePairs is there because there appear to be some problems with the language at the moment. Need to regenerate pairs from singles
"""
def getLanguage(modelInfo, modelTypeDir,
				regeneratePairs = True,
				verbose = False):
	# Sets up where to get information about the language from
	langDirTrials = modelTypeDir + 'Languages/Lang_' + str(int(modelInfo['seed'])) + '/Trials.csv'
	langDirInfo = modelTypeDir + 'Languages/Lang_' + str(int(modelInfo['seed'])) + '/Info.csv'
	langInfo = {'seed':int(modelInfo['seed'])}
	# Extracts information about the language. We care about the singles training trials
	singles = pandas.read_csv(langDirTrials)
	singles = singles.loc[singles['numWords'] == 1]
	# Next we want to generate all possible testing pairs from the training set
	# Therefore, we are going to have len(trial_training_singles)^2 trials
	testingTrials = []
	for index1, trial1 in singles.iterrows():
		for index2, trial2 in singles.iterrows():
			testingTrials.append(testingTrial(trial1, trial2))
	return testingTrials

if __name__ == '__main__':
	verbose = True
	if verbose:
		np.set_printoptions(precision = 2, suppress = True)
	modelTypes = getChildren(direcs = os.getcwd() + "/ModelTypes/*/",
						 	 return_files = False)
	for modelType in modelTypes:
		file = os.getcwd() + '/PredAnalysis/' + modelType.split('/')[-2] + "_AllModels.csv"
		sys.path.append(modelType)
		# Imports (or reimports) generateModels and generateLanguage for each kind of model
		if 'generateModels' in sys.modules:
			importlib.reload(generateModels)
		else:
			import generateModels
		# Get log of models that contains model information which will be used to recreate model
		modelLog = pandas.read_csv(modelType + 'Models/Model_Log.csv')
		# Get language for each model type
		# Iterate through each model, create model, and examine model predictions
		for model in modelLog['modelName'].tolist():
			modelRow = modelLog.loc[modelLog['modelName'] == model]
			modelInfo = {'modelType':modelType,
						 'modelName':model, 
						 'modelSeed':int(modelRow['modelSeed']), 
						 'trainingSeed':int(modelRow['trainingSeed']), 
						 'seed':int(modelRow['seed']), 
						 'learning_rate':float(modelRow['learning_rate']), 
						 'num_trials':int(modelRow['num_trials']), 
						 'num_epochs':int(modelRow['num_epochs']), 
						 'num_timeSteps':int(modelRow['num_timeSteps']), 
			 			 'num_batches':int(modelRow['num_batches']), 
			 			 'size_layer_message':int(modelRow['size_layer_message']), 
			 			 'size_layer_hidden':int(modelRow['size_layer_hidden']), 
			 			 'size_layer_phonology':int(modelRow['size_layer_phonology'])}
			testingTrials = getLanguage(modelInfo = modelInfo, 
										modelTypeDir = modelType,
										verbose = verbose)
			buildModel(modelInfo = modelInfo,
					   testingTrials = testingTrials,
					   file = file,
					   verbose = verbose)