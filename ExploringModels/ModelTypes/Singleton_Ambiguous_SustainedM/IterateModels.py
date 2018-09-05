import os
import random
import generateLanguage
import generateModels

"""
Creates a log of all models trained
"""
def modelTracker(modelLog, langInfo, modelInfo, modelName,
				 header = ['modelName', 'modelSeed', 'trainingSeed', 'seed', 'learning_rate', 'num_trials', 'num_epochs', 'num_timeSteps', 'num_batches', 'size_layer_message', 'size_layer_hidden', 'size_layer_phonology', 'input', 'hidden_state_init', 'hidden_state_curr', 'predictions', 'max_length_name'],
				 verbose = False):
	langInfo.update(modelInfo)
	langInfo['modelName'] = modelName
	generateLanguage.recordResponse(fileName = modelLog, response = langInfo, header = header)
	if verbose:
		print("\tModel {} logged".format(modelInfo['modelName']))

"""
Generates and trains multiple models. Saves outputs of each model.
"""
def iterateModels(l_learning_rate, l_num_trials, l_num_epochs, l_num_timeSteps, l_num_batches, l_size_layer_hidden, size_layer_message, size_layer_phonology, modelTrackerFile, langInfo, modelSeed, trainingSeed,
				  verbose = False):
	modelNames = []
	modelInfo = {'size_layer_message':size_layer_message,
				 'size_layer_phonology':size_layer_phonology,
				 'modelSeed':modelSeed,
				 'trainingSeed':trainingSeed}
	# Iterate through the different parameters
	if verbose:
		numModels = len(l_learning_rate)*len(l_num_trials)*len(l_num_epochs)*len(l_num_timeSteps)*len(l_num_batches)*len(l_size_layer_hidden)
		print("Beginning iterative training of {} models".format(numModels))
	for learning_rate in l_learning_rate:
		modelInfo['learning_rate'] = learning_rate
		for num_trials in l_num_trials:
			modelInfo['num_trials'] = num_trials
			for num_epochs in l_num_epochs:
				modelInfo['num_epochs'] = num_epochs
				for num_timeSteps in l_num_timeSteps:
					modelInfo['num_timeSteps'] = num_timeSteps
					for num_batches in l_num_batches:
						modelInfo['num_batches'] = num_batches
						for size_layer_hidden in l_size_layer_hidden:
							modelInfo['size_layer_hidden'] = size_layer_hidden
							if verbose:
								print("Remaining models: {}".format(numModels))
							# Set up new model
							graph, nodes, modelInfo = generateModels.generateModel(modelInfo = modelInfo,
				  		  		 						 						   verbose = verbose)
							# Generate a random identifier for the model
							while True:
								modelName = random.randint(100000000, 999999999)
								if modelName not in modelNames:
									modelNames.append(modelName)
									break
							modelInfo['modelName'] = modelName
							# Create locations to save model information
							saveGraphLocation = os.getcwd() + "/Models/Model" + str(modelName)
							saveVariablesLocation = os.getcwd() + "/Models/Model" + str(modelName) + "/"
							saveLossesLocation = os.getcwd() + "/Models/Model" + str(modelName) + "/ModelLosses.csv"
							# Train model
							training_losses = generateModels.trainNetwork(modelInfo = modelInfo,
																		  graph = graph,
																		  nodes = nodes,
																		  trials_training_singles = trials_training_singles,
														  				  saveVariablesLocation = saveVariablesLocation,
														  				  saveLossesLocation = saveLossesLocation,
														  				  verbose = verbose)
							# Save meta-information about the model
							modelTracker(modelLog = modelTrackerFile,
										 langInfo = langInfo, 
										 modelInfo = modelInfo, 
										 modelName = modelName)
							if verbose:
								numModels -= 1

if __name__ == '__main__':
	verbose = True
	# Read in trials
	langInfo = {'seed' : 896575376869}
	pathName = os.getcwd() + '/Languages/Lang_' + str(langInfo['seed'])
	langDirTrials = pathName + '/Trials.csv'
	langDirInfo = pathName + '/Info.csv'
	trials_training_singles, langInfo = generateLanguage.regenerateFromFile(langDirTrials = langDirTrials, langDirInfo = langDirInfo, langInfo = langInfo, verbose = verbose)
	# Create models with different parameters
	modelSeed = 9021093912
	trainingSeed = 927986397102
	l_learning_rate = [0.005]
	l_num_trials = [20000] # Number of trials per epoch
	l_num_epochs = [100] # Number of groups of trials
	l_num_timeSteps = [6, 7, 8, 9, 10] # Number of time steps over which time is calculated
	l_num_batches = [3, 5, 10]  # Number of batches into which we divide our data
	l_size_layer_hidden = [50] # Size of hidden layer
	size_layer_message = len(trials_training_singles)
	size_layer_phonology = langInfo['numPhon']
	modelTrackerFile = os.getcwd() + "/Models/Model_Log.csv"
	random.seed(langInfo['seed'])
	iterateModels(l_learning_rate = l_learning_rate,
				  l_num_trials = l_num_trials,
				  l_num_epochs = l_num_epochs,
				  l_num_timeSteps = l_num_timeSteps,
				  l_num_batches = l_num_batches,
				  l_size_layer_hidden = l_size_layer_hidden,
				  size_layer_message = size_layer_message,
				  size_layer_phonology = size_layer_phonology,
				  modelTrackerFile = modelTrackerFile,
				  langInfo = langInfo,
				  modelSeed = modelSeed,
				  trainingSeed = trainingSeed,
				  verbose = verbose)