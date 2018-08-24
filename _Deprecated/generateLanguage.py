import random
import os
import glob
import numpy as np

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

if __name__ == '__main__':
	seed = 896575376869
	random.seed(seed)
	verbose = True
	# Start by generating the stimuli used for the model
	if verbose:
		print("Random seed: {}".format(seed))
	numTargets = 20 # Ge10nerally, keep this number below 500, as may start causing difficulties finding words w/ random
	numPhon = 18
	lenPhon_Total = [3]
	numAmbig = 0
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
	 								 		numPhon = numPhon,
	 								 		lenPhon_Interfering = lenPhon_Interfering,
	 								 		words = words,
	 								 		verbose = verbose)
	testReserve = 0.10
	trials_training_singles, trials_training_pairs, trials_testing = generateTrials(messages_target = messages_target,
					 								 								messages_primes = messages_primes,
					 								 								lenPhon_Interfering = lenPhon_Interfering,
					 								 								testReserve = testReserve,
					 								 								verbose = verbose)