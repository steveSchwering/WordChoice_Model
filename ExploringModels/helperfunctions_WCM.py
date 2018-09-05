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
				record += str(response.get(value)) + separator
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
				record += str(response.get(value)) + separator
			record = record[:-len(separator)]
			record += ender
			f.write(record)