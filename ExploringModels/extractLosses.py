import pandas
import os
import glob
from helperfunctions_WCM import *

"""
Get all models within the type and all of the model .csv files and then extracts losses
"""
def getLosses(modelType,
			  verbose = False):
	modelDirecs = getChildren(direcs = modelType + "Models/*/", return_files = False)
	allLosses = {}
	for modelDirec in modelDirecs:
		modelLosses = pandas.read_csv(getChildren(files = modelDirec + "/*", filetype = '.csv', return_direcs = False)[0])
		allLosses[modelLosses['modelName'].tolist()[0]] = modelLosses['loss'].tolist()
		if verbose:
			print("\tRetrieved model: {}".format(modelLosses['modelName']))
	return allLosses

if __name__ == '__main__':
	verbose = True
	modelTypes = getChildren(direcs = os.getcwd() + "/ModelTypes/*/",
						 	 return_files = False)
	for modelType in modelTypes:
		filename = os.getcwd() + '/LossAnalysis/' + modelType.split('/')[-2] + '_AllModels_LossAnalysis' + '.csv'
		if verbose:
			print("Starting: {}".format(filename.split('/')[-1]))
		# Gets all model info
		try:
			modelLog = pandas.read_csv(modelType + 'Models/Model_Log.csv')
		except:
			print("Error retrieving model log of type: {}".format(modelType))
			print("\tMoving on to new model type")
			continue
		# Gets all model names
		models = modelLog['modelName'].tolist()
		# Gets all model losses
		allLosses = getLosses(modelType, verbose)
		# Combines this infomration
		header = ['modelName', 
			 		   'modelSeed', 
			 		   'trainingSeed', 
			 		   'seed', 
			 		   'learning_rate', 
			 		   'num_trials', 
			 		   'num_epochs', 
			 		   'num_timeSteps', 
			 		   'num_batches', 
			 		   'size_layer_message', 
			 		   'size_layer_hidden', 
			 		   'size_layer_phonology',
			 		   'epoch',
			 		   'loss']
		# Outputs this information into a new excel file
		for model in models:
			modelRow = modelLog.loc[modelLog['modelName'] == model]
			modelInfo = {'modelName':model, 
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
			for index, loss in enumerate(allLosses[model]):
				modelInfo['epoch'] = index
				modelInfo['loss'] = loss
				recordResponse(fileName = filename,
							   response = modelInfo,
							   header = header)