## Util functions for ML models
# Shway Wang
# Nov. 1, 2022

## Imports
from math import floor
import numpy as np
from jax import random as jrandom

## function to initialize the parameters randomly
def random_params_by_size(n, m, key=jrandom.PRNGKey(0), scale=1e-2):
	if (m is None):
		return scale * jrandom.normal(key, (n,))
	elif (n is None):
		return scale * jrandom.normal(key, (m,))
	return scale * jrandom.normal(key, (n, m))

## function to get unique tokens from the dataset
def getTokens(lines: list) -> list:
	tokens = []
	for line in lines:
		for t in line:
			if (t not in tokens):
				tokens.append(t)
	return tokens

## function to convert data to one hot vectors
def toOneHot(lines: list, tokens: list) -> list:
    oneHot = []
    for lineI in range(len(lines)):
        line = lines[lineI]
        numedLine = np.array([tokens.index(token) for token in line])
        lineVec = np.zeros((numedLine.size, len(tokens)))
        lineVec[np.arange(numedLine.size), numedLine] = 1
        oneHot.append(lineVec)
    return oneHot

def getMaxSeqLen(lines: list) -> int:
	maxSeqLen = 0
	for lineI in range(len(lines)):
		if (len(lines[lineI]) > maxSeqLen):
			maxSeqLen = len(lines[lineI])
	return maxSeqLen

# output vector to string transformation
def vec2str(vec, tokens):
	return tokens[np.argmax(vec, axis = 0)]

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

def levelDataPreProc(path):
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
