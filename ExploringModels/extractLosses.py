import pandas
import os
import glob
from helperfunctions_WCM import *

"""
Get all models within the type and all of the model .csv files and then extracts losses
"""
def getLosses(modelType):
	modelDirecs = getChildren(direcs = modelType + "Models/*/", return_files = False)
	allLosses = {}
	for modelDirec in modelDirecs:
		modelLosses = pandas.read_csv(getChildren(files = modelDirec + "/*", filetype = '.csv', return_direcs = False)[0])
		allLosses[modelLosses['modelName'].tolist()[0]] = modelLosses['loss'].tolist()
	return allLosses

if __name__ == '__main__':
	modelTypes = getChildren(direcs = os.getcwd() + "/ModelTypes/*/",
						 	 return_files = False)
	for modelType in modelTypes:
		filename = os.getcwd() + '/LossAnalysis/' + modelType.split('/')[-2] + '_AllModels' + '.csv'
		print(filename)
		# Gets all model info
		modelLog = pandas.read_csv(modelType + 'Models/Model_Log.csv')
		# Gets all model names
		models = modelLog['modelName'].tolist()
		# Gets all model losses
		allLosses = getLosses(modelType)
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