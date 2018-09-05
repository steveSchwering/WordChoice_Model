import glob
import os
import pandas
import sys
import importlib
from scipy import spatial
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
		
	def getInfo(self):
		outdict = {'word1_type':self.word1_type,
				   'word1_message':self.word1_message,
				   'word1_label':self.word1_label,
				   'word1_phonemes':self.word1_phonemes,
				   'word1_ambiguous':self.word1_ambiguous,
				   'word2_type':self.word2_type,
				   'word2_message':self.word2_message,
				   'word2_label':self.word2_label,
				   'word2_phonemes':self.word2_phonemes,
				   'word2_ambiguous':self.word2_ambiguous,
				   'matching_message':self.matchingMessage}
		phon_keys = []
		for i, mp in enumerate(self.matchingPhonology):
			key = 'matching_phonology' + str(i)
			outdict.update({key:mp})
			phon_keys.append(key)
		return outdict, phon_keys


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
		yield(x, y_hat, trial, num_timeSteps)

"""
Actually saves the predictions
"""
def savePredictions(modelInfo, trial, X, Y_hat, predictions, filename):
	response = modelInfo
	trialDict, phonKeys = trial.getInfo()
	response.update(trialDict)
	response['X_total'] = X
	response['Y_hat_total'] = Y_hat
	response['predictions_total'] = predictions
	n_values = int(modelInfo['size_layer_phonology'])
	header = ['modelType',
			  'epoch',
   			  'modelName',
   			  #'modelSeed',
   			  #'trainingSeed',
   			  #'seed',
   			  #'learning_rate',
   			  #'num_trials',
   			  #'num_epochs',
   			  'num_timeSteps',
   			  'num_batches',
   			  #'size_layer_message',
   			  #'size_layer_hidden',
   			  #'size_layer_phonology',
   			  'word1_type',
   			  'word1_message',
   			  #'word1_label',
   			  'word1_ambiguous',
   			  'word2_type',
   			  'word2_message',
   			  #'word2_label',
   			  'word2_ambiguous',
   			  'matching_message']
	header += phonKeys
	header += ['time',
   			   'cosine_sim',
   			   'target_value']
	for timeStep, (y_hat, prediction) in enumerate(zip(Y_hat.flatten(), predictions[0])):
		y_hat_onehot = (np.eye(n_values)[y_hat])
		"""print("Expected: \n{}".format(y_hat_onehot))
		print("Predicted: \n{}\n".format(prediction))"""
		response['cosine_sim'] = (1 - spatial.distance.cosine(y_hat_onehot, prediction))
		response['target_value'] = prediction[y_hat]
		response['time'] = timeStep
		recordResponse(fileName = filename,
				  	   response = response,
				  	   header = header)

"""
Run model and see what we get
"""
def getPredictions(modelInfo, graph, nodes, testingTrials, file,
				   limitModels = True,
				   verbose = False):
	# gets predictions from what model produces
	graphLoc = modelInfo['modelType'] + "Models/Model" + str(modelInfo['modelName']) + "/"
	if verbose:
		print("Predicting for model {}".format(graphLoc))
	graphs = glob.glob(graphLoc + "*.meta")
	if limitModels:
		graphs = [graphs[-1]]
	for graph in graphs:
		if verbose:
			print("\tPredicting for graph saved: {}".format(graph.split('/')[-1]))
		data=graph.split('.')[0]
		modelInfo['epoch'] = graph.split('-')[-1].split('.')[0]
		tf.reset_default_graph()
		x = tf.placeholder(tf.int32,[None, None], name = 'Message')
		y = tf.placeholder(tf.int32, [None, None], name = 'Phonology')
		with tf.Session() as sess:
			restoredGraph = tf.train.import_meta_graph(graph)
			restoredGraph.restore(sess, data)
			for i, test in enumerate(generateTest(trials = testingTrials)):
				training_state = np.zeros((1, modelInfo['size_layer_hidden']))
				(X, Y, trial, num_timeSteps) = test
				predictions, training_state = sess.run([modelInfo['predictions'], modelInfo['hidden_state_curr']], feed_dict={modelInfo['input']:X, modelInfo['hidden_state_init']:training_state, modelInfo['max_length_name']:num_timeSteps})
				savePredictions(modelInfo = modelInfo, 
								trial = trial,
								X = X,
								Y_hat = Y,
								predictions = predictions,
								filename = file)

"""
Regenerates model from 
"""
def buildModel(modelInfo, testingTrials, file,
			   limitModels = True,
			   verbose = False):
	# reconstructs model from tensorflow
	tf.reset_default_graph()
	try:
		graph, nodes, modelInfo = generateModels.generateModel(modelInfo = modelInfo,
															   verbose = verbose)
	except:
		print("Error recreating model {} of model type {}".format(modelInfo['modelName'], modelType.split('/')[-2]))
		return
	# Run model recording predictions
	getPredictions(modelInfo = modelInfo,
				   graph = graph,
				   nodes = nodes,
				   testingTrials = testingTrials,
				   file = file,
				   limitModels = limitModels,
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
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	verbose = True
	limitModels = True # TURN THIS ON TO ENSURE WE ONLY RETURN THE PREDICTIONS FROM THE LAST EPOCH OF TRAINING
	if verbose:
		np.set_printoptions(precision = 2, suppress = True)
	modelTypes = getChildren(direcs = os.getcwd() + "/ModelTypes/*/",
						 	 return_files = False)
	# We will analyze each type of model separately
	for modelType in modelTypes:
		file = os.getcwd() + '/PredAnalysis/' + modelType.split('/')[-2] + "_Predictions_AllModels.csv"
		if verbose:
			print("Starting to add to {}".format(file))
		sys.path.append(modelType)
		# Imports (or reimports) generateModels and generateLanguage for each kind of model
		if 'generateModels' in sys.modules:
			importlib.reload(generateModels)
		else:
			import generateModels
		# Get log of models that contains model information which will be used to recreate model
		try:
			modelLog = pandas.read_csv(modelType + 'Models/Model_Log.csv')
		except:
			print("Error retrieving model log of type: {}".format(modelType))
			print("\tMoving on to new model type")
			continue
		# Get language for each model type
		# Iterate through each model, create model, and examine model predictions
		for model in modelLog['modelName'].tolist():
			modelRow = modelLog.loc[modelLog['modelName'] == model]
			# Generating model information
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
			# Generating language
			testingTrials = getLanguage(modelInfo = modelInfo, 
										modelTypeDir = modelType,
										verbose = verbose)
			# Building and analyzing model
			buildModel(modelInfo = modelInfo,
					   testingTrials = testingTrials,
					   file = file,
					   limitModels = limitModels,
					   verbose = verbose)