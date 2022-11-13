#### Util functions for ML models

## Imports
from math import floor
import numpy as np
from jax import random as jrandom

## function to initialize the parameters randomly
# Shway Wang
# Nov. 1, 2022
def random_params_by_size(n, m, key=jrandom.PRNGKey(0), scale=1e-2):
	if (m is None):
		return scale * jrandom.normal(key, (n,))
	elif (n is None):
		return scale * jrandom.normal(key, (m,))
	return scale * jrandom.normal(key, (n, m))

## function to get unique tokens from the dataset
# Shway Wang
# Nov. 1, 2022
def getTokens(lines: list) -> list:
	tokens = []
	for line in lines:
		for t in line:
			if (t not in tokens):
				tokens.append(t)
	return tokens

## function to convert data to one hot vectors
# Shway Wang
# Nov. 1, 2022
def toOneHot(lines: list, tokens: list) -> list:
    oneHot = []
    for lineI in range(len(lines)):
        line = lines[lineI]
        numedLine = np.array([tokens.index(token) for token in line])
        lineVec = np.zeros((numedLine.size, len(tokens)))
        lineVec[np.arange(numedLine.size), numedLine] = 1
        oneHot.append(lineVec)
    return oneHot

## gets the maximum length of the data sequence
# Shway Wang
# Nov. 1, 2022
def getMaxSeqLen(lines: list) -> int:
	maxSeqLen = 0
	for lineI in range(len(lines)):
		if (len(lines[lineI]) > maxSeqLen):
			maxSeqLen = len(lines[lineI])
	return maxSeqLen

## output vector to string transformation
# Shway Wang
# Nov. 1, 2022
def vec2str(vec, tokens):
	return tokens[np.argmax(vec, axis = 0)]
	
## output vector to level tile transformation
# Shway Wang
# Nov. 12, 2022
def vec2tile(vec, tokens):
	ai = np.argmax(vec[:-2], axis = 0)
	bi = np.argmax(vec[-2:], axis = 0)
	return tokens[ai], tokens[len(vec[:-2]) + bi]

## preprocesses the text data
# Shway Wang
# Nov. 1, 2022
def textDataPreProc(path):
	# get all the lines
	lines = []
	path += '{}'
	for i in range(1, 402):
	    if (i < 10):
	        index = "00{}.txt".format(i)
	    elif (i < 100):
	        index = "0{}.txt".format(i)
	    else:
	        index = "{}.txt".format(i)
	    with open(path.format(index), 'r') as f:
	        lines.extend(f.readlines())

	# Remove new lines and empty lines
	lines = [line for line in lines if line != '\n' or '']

	# Strip all lines
	lines = [line.strip() for line in lines]

	# remove lines that are less than 10 tokens
	lines = [line for line in lines if len(line) > 10]

	seqMaxLen = getMaxSeqLen(lines)

	# get the unique tokens
	tokens = getTokens(lines)

	# split the lines to training set and test set
	splitInd = floor(0.8 * len(lines))
	train = lines[:splitInd + 1]
	test = lines[splitInd + 1:]

	# convert the training set and test set into one hot vectors
	trainVec = toOneHot(train, tokens)
	testVec = toOneHot(test, tokens)
	
	# return the one hot vectors of training and test sets and the set of tokens
	return trainVec, testVec, tokens, seqMaxLen
	
# Process the level and path data into a form that can be used by the LSTM 
# Justin Stevens
# modified by Shway Wang
def processPathAndLevel():
	levelData = open("Mario-AI-Framework/src/levels/original/lvl-1sa.txt", "r")
	pathData = open("Mario-AI-Framework/src/Data/Justin/lvl-1sa.txt", "r")

	levelLines = levelData.readlines()

	pathLines = pathData.readlines()

	tokens = getTokens(levelLines)
	oneHotLine = toOneHot(levelLines, tokens)
	
	pathTokens = getTokens(pathLines)
	oneHotPath = toOneHot(pathLines, pathTokens)

	tokens.extend(pathTokens)
	
	# the combined lines
	lines = [np.hstack([oneHotLine[0][:100], oneHotPath[0][:100]])]
	
	# get the maximum sequence length
	seqMaxLen = getMaxSeqLen(lines)
	
	return lines, tokens, seqMaxLen
	
	
	
