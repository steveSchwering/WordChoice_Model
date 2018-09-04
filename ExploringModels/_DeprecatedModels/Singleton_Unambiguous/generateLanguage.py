import random
import os
import glob
import pandas
import ast
import numpy as np

class Trial():
	def __init__(self, numWords, condition, image1, label1, phonemes1, image2, label2, phonemes2):
		self.numWords = numWords
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
	def get_message_np(self):
		zeros = np.array([0]*(len(self.phonemes1)), dtype = np.float32)
		if self.image2 == None:
			zeros[0] = int(self.image1)
			return zeros.astype(np.int32)
		else:
			return out1, out2
		
	"""
	Outputs one-hot vectors of model output
	"""
	def get_phonology_np(self):
		phonemes1 = [int(_) for _ in self.phonemes1]
		if self.image2 == None:
			return np.asarray(self.phonemes1).astype(np.int32)
		else:
			phonemes2 = [int(_) for _ in self.phonemes2]
			return np.asarray(phonemes1).astype(np.int32), np.asarray(phonemes2).astype(np.int32)

"""
This function generates message:word pairings in a dictionary format
+ numTargets tells us how many 'messages' there are
+ 	Remember a single message can have multiple words words associated with it (e.g. 'jacket' or 'coat')
+ numPhon tells us how many phonological units there are
+ lenPhon_Total tells us how many phonological units are associated with each word
+ numAmbigTargets tells us how many of the messages have multiple possible words associated with it
+ 	We want some words to be ambiguous and some to not be ambiguous, so this is usually set to half of numTargets
"""
def generateTargetMessages(numTargets, numPhon, lenPhon_Total, numAmbigTargets,
						   verbose = False):
	assert numAmbigTargets <= numTargets
	# Which messages are going to have to have two words associated with it?
	shared = ([1] * int(numAmbigTargets)) + ([0] * (int(numTargets) - int(numAmbigTargets)))
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
+ numInterfering indicates the num of messages_target pairs that will have interfering message primes
+ 	Default should be all of them
+ lenPhon_Total indicates possible word lengths of the language
+ lenPhon_Interfering indicates how may of the elements of the phonological list starting from index 0 are overlapping between prime and target
+ words keeps track of what words were already created. Passed in from prior function
"""
def generatePrimes(messages_target, numInterfering, lenPhon_Total, numPhon, lenPhon_Interfering, words,
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
			messages_primes[primeNum] = []
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
			if len(messages_target[message]) > 1:
				trials_ambiguous.append(Trial(numWords = 1, condition = 'ambiguous', image1 = message, label1 = label, phonemes1 = phonological_code, image2 = None, label2 = None, phonemes2 = None))
			else:
				trials_ambiguous.append(Trial(numWords = 1, condition = 'unambiguous', image1 = message, label1 = label, phonemes1 = phonological_code, image2 = None, label2 = None, phonemes2 = None))
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
			trials_unambiguous.append(Trial(numWords = 1, condition = 'unambiguous', image1 = messageNum, label1 = label, phonemes1 = phonological_code, image2 = None, label2 = None, phonemes2 = None))
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
						trials_interfering.append(Trial(numWords = 2, condition = 'interfering', image1 = message_prime, label1 = label1, phonemes1 = m_p, image2 = message_target, label2 = label2, phonemes2 = m_t))
					else:
						trials_noninterfering.append(Trial(numWords = 2, condition = 'noninterfering', image1 = message_prime, label1 = label1, phonemes1 = m_p, image2 = message_target, label2 = label2, phonemes2 = m_t))
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
Organizes a language for reading trials into a csv file
"""
def saveLanguage(singles, pairs, testingPairs, langDirTrials, langDirInfo, langInfo,
				 headerWords = ['numWords', 'train', 'condition', 'image1', 'label1', 'phonemes1', 'image2', 'label2', 'phonemes2'],
				 headerInfo = ['seed', 'numTargets', 'numAmbigTargets', 'numInterfering', 'numPhon', 'lenPhon_Total', 'lenPhon_Interfering', 'testReserve']):
	output = {}
	for single in singles:
		output['numWords'] = single.numWords
		output['condition'] = single.condition
		output['image1'] = single.image1
		output['label1'] = single.label1
		output['phonemes1'] = '_'.join(single.phonemes1)
		output['image2'] = single.image2
		output['label2'] = single.label2
		if single.phonemes2:
			output['phonemes2'] = '_'.join(single.phonemes2)
		else:
			output['phonemes2'] = None
		output['train'] = 'train'
		recordResponse(fileName = langDirTrials, response = output, header = headerWords)
	for pair in pairs:
		output['numWords'] = pair.numWords
		output['condition'] = pair.condition
		output['image1'] = pair.image1
		output['label1'] = pair.label1
		output['phonemes1'] = str('_'.join(pair.phonemes1))
		output['image2'] = pair.image2
		output['label2'] = pair.label2
		if pair.phonemes2:
			output['phonemes2'] = str('_'.join(pair.phonemes1))
		else:
			output['phonemes2'] = None
		output['train'] = 'train'
		recordResponse(fileName = langDirTrials, response = output, header = headerWords)
	for pair in testingPairs:
		output['numWords'] = pair.numWords
		output['condition'] = pair.condition
		output['image1'] = pair.image1
		output['label1'] = pair.label1
		output['phonemes1'] = '_'.join(pair.phonemes1)
		output['image2'] = pair.image2
		output['label2'] = pair.label2
		if pair.phonemes2:
			output['phonemes2'] = str('_'.join(pair.phonemes1))
		else:
			output['phonemes2'] = None
		output['train'] = 'test'
		recordResponse(fileName = langDirTrials, response = output, header = headerWords)
	langInfo['lenPhon_Total'] = str('_'.join([str(_) for _ in langInfo['lenPhon_Total']]))
	recordResponse(fileName = langDirInfo, response = langInfo, header = headerInfo)

"""
Records the language into a csv file
"""
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

def regenerateFromFile(langDirTrials, langDirInfo, langInfo,
					   verbose = False):
	assert '.csv' in langDirTrials
	assert '.csv' in langDirInfo
	allTrials = pandas.read_csv(langDirTrials)
	allTrials = allTrials.replace({'None':None})
	trials_training_singles = []
	trials_training_pairs = []
	trials_testing = []
	if verbose:
		print("Regenerating language from {}".format(langDirTrials))
	for index, row in allTrials.iterrows():
		if verbose:
			print("\tRegenerating trial {} -- ".format(index), end = '')
		numWords = int(row['numWords'])
		condition = row['condition']
		image1 = int(row['image1'])
		label1 = row['label1']
		phonemes1 = row['phonemes1'].split('_')
		if numWords == 1:
			image2 = None
			label2 = None
			phonemes2 = None
		elif numWords == 2:
			image2 = int(row['image2'])
			label2 = row['label2']
			phonemes2 = row['phonemes2'].split('_')
		trial = Trial(numWords = numWords, 
					  condition = condition, 
					  image1 = image1, 
					  label1 = label1, 
					  phonemes1 = phonemes1, 
					  image2 = image2, 
					  label2 = label2, 
					  phonemes2 = phonemes2)
		if (int(row['numWords']) == 1):
			trials_training_singles.append(trial)
			if verbose:
				print("single {}".format(len(trials_training_singles)))
		elif ((int(row['numWords']) == 2) & (row['train'] == 'train')):
			trials_training_pairs.append(trial)
			if verbose:
				print("pair for training {}".format(len(trials_training_pairs)))
		elif ((int(row['numWords']) == 2) & (row['train'] == 'test')):
			trials_testing.append(trial)
			if verbose:
				print("pair for testing {}".format(len(trials_testing)))
	if verbose:
		print("Regenerating language information from {}".format(langDirInfo))
	info = pandas.read_csv(langDirInfo)
	langInfo.update({'numTargets' : int(info['numTargets'].iloc[0]),
					 'numPhon' : int(info['numPhon'].iloc[0]),
					 'lenPhon_Total' : [int(_) for _ in (str(info['lenPhon_Total'].iloc[0])).split('_')],
					 'numAmbigTargets' : int(info['numAmbigTargets'].iloc[0]),
					 'numInterfering' : int(info['numInterfering'].iloc[0]),
					 'lenPhon_Interfering' : int(info['lenPhon_Interfering'].iloc[0]),
					 'testReserve' : int(info['testReserve'].iloc[0])})
	if verbose:
		for key in langInfo:
			print("\t{} : {}".format(key, langInfo[key]))
	return trials_training_singles, trials_training_pairs, trials_testing, langInfo

if __name__ == '__main__':
	verbose = True
	save = True
	langInfo = {'seed' : 896575376869}
	pathName = os.getcwd() + '/Languages/Lang_' + str(langInfo['seed'])
	langDirTrials = pathName + '/Trials.csv'
	langDirInfo = pathName + '/Info.csv'
	if not os.path.isfile(langDirTrials):
		if not os.path.exists(pathName):
			os.makedirs(pathName)
		langInfo.update({'numTargets' : 40, # Number of messages that CAN have multiple words associated with them
						 'numPhon' : 18, # Number of phonemes in the language
						 'lenPhon_Total' : [3], # Possible word lengths
						 'numAmbigTargets' : 0, # Number of targets that are 'ambiguous' -- the number of messages that DO have multiple words associated with them
						 'numInterfering' : 0, # Number of target messages that have interfering words associated with them
						 'lenPhon_Interfering' : 2, # Length of the phonological interference})
						 'testReserve' : 0.10})
		random.seed(langInfo['seed'])
		# Start by generating the stimuli used for the model
		if verbose:
			print("Random seed: {}".format(langInfo['seed']))
		words, messages_target = generateTargetMessages(numTargets = langInfo['numTargets'], 
														numPhon = langInfo['numPhon'],
										   				lenPhon_Total = langInfo['lenPhon_Total'],
								   		   				numAmbigTargets = langInfo['numAmbigTargets'],
								   		   				verbose = verbose)
		words, messages_primes = generatePrimes(messages_target = messages_target,
		 								 		numInterfering = langInfo['numInterfering'],
		 								 		lenPhon_Total = langInfo['lenPhon_Total'],
		 								 		numPhon = langInfo['numPhon'],
		 								 		lenPhon_Interfering = langInfo['lenPhon_Interfering'],
		 								 		words = words,
		 								 		verbose = verbose)
		trials_training_singles, trials_training_pairs, trials_testing = generateTrials(messages_target = messages_target,
						 								 								messages_primes = messages_primes,
						 								 								lenPhon_Interfering = langInfo['lenPhon_Interfering'],
						 								 								testReserve = langInfo['testReserve'],
						 								 								verbose = verbose)
		if save:
			saveLanguage(singles = trials_training_singles, pairs = trials_training_pairs, testingPairs = trials_testing, langDirTrials = langDirTrials, langDirInfo = langDirInfo, langInfo = langInfo)
	else:
		trials_training_singles, trials_training_pairs, trials_testing, langInfo = regenerateFromFile(langDirTrials = langDirTrials, langDirInfo = langDirInfo, langInfo = langInfo, verbose = verbose)