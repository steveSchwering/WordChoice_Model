import pandas
import os
import glob

"""
Returns all children directores and files separated into different lists
+ Options to return only a subset of either directories or files and a subset of a specific kind of files
+ Defaults to returning both all directories and all files
"""
def getChildren(direcs = None, files = None, filetype = None, return_direcs = True, return_files = True):
	if direcs == None:
		direcs = os.getcwd() + "/*/"
	if files == None:
		files = os.getcwd() + "/*"
	if filetype != None:
		files += filetype
	if return_files & return_direcs:
		return glob.glob(direcs), glob.glob(files)
	elif return_direcs:
		return glob.glob(direcs)
	elif return_files:
		return glob.glob(files)
	else:
		return None

def recordResponse(fileName, response, header,
				   separator = ',',
				   ender = "\n"):
	if os.path.exists(fileName):
		writeCode = 'a'
		with open(fileName, writeCode) as f:
			record = ""
			for value in header:
				record += str(response[value]) + separator
			record = record[:-len(separator)]
			record += ender
			f.write(record)
	else:
		writeCode = 'w'
		with open(fileName, writeCode) as f:
			record = ""
			for variable in header:
				record += variable + separator
			record = record[:-len(separator)]
			record += ender
			f.write(record)
			record = ""
			for value in header:
				record += str(response[value]) + separator
			record = record[:-len(separator)]
			record += ender
			f.write(record)

def getLosses(modelType):
	# Get all models within the type and all of the model .csv files
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