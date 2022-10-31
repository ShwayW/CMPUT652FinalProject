## Imports
# jax related
import jax.numpy as jnp
import jax.nn as jnn
from jax import grad, jit, vmap, lax, value_and_grad
from jax import random as jrandom
from jax.scipy.special import logsumexp

# others
from math import floor
import numpy as np
import os
import pandas as pd
import opendatasets as od
import time
from random import randint
import pickle

## Download the dataset
############ DO NOT PUSH DATASET TO GITHUB #################
#od.download("https://www.kaggle.com/datasets/pariza/bbc-news-summary")

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

## Implementation of components of LSTM
# initialize the parameters randomly
def random_params_by_size(n, m, key, scale=1e-2):
    if (m is None):
        return scale * jrandom.normal(key, (n,))
    return scale * jrandom.normal(key, (n, m))

# an individual lstm cell
def lstm_cell(params, prevCell, prevHidden, curToken):
	# assumptions:
	# params[0] is w_f, b_f
	# params[1] is w_i, b_i
	# params[2] is w_c, b_c
	# params[3] is w_o, b_o

	combined = prevHidden + curToken
	f = jnn.sigmoid(jnp.dot(params[0][0], combined) + params[0][1])
	i = jnn.sigmoid(jnp.dot(params[1][0], combined) + params[1][1])
	cand = jnp.tanh(jnp.dot(params[2][0], combined) + params[2][1])
	o = jnn.sigmoid(jnp.dot(params[3][0], combined) + params[3][1])
	curCell = jnp.multiply(f, prevCell) + jnp.multiply(i, cand)
	curHidden = jnp.multiply(o, jnp.tanh(curCell))
	#print(combined, curCell, curHidden)
	#print()
	return curCell, curHidden

jitLstmCell = jit(lstm_cell)

# a sequence of lstm cells
@jit
def lstm_seq(params, prevCell, prevHidden, curInput):
	assert(len(params) == len(curInput))
	for inputI in range(len(curInput)):
		prevCell, prevHidden = jitLstmCell(params[inputI], prevCell,
			prevHidden, curInput[inputI])
	return prevCell, prevHidden

## Implementation of the loss and update functions
# loss of lstm cells on a given input sequence predicting the next token
@jit
def lstm_seq_loss(params, prevCell, prevHidden, curInput, targetOutput):
	# curInput is a sequence
	# targetOutput is a single token
	assert(len(params) == len(curInput))
	for inputI in range(len(curInput)):
		prevCell, prevHidden = jitLstmCell(params[inputI], prevCell,
			prevHidden, curInput[inputI])
	return jnp.mean(jnp.absolute(prevHidden - targetOutput))

# lstm sequence update
@jit
def lstm_seq_update(params, grads, stepSize):
	for paramI in range(len(params)):
		param = params[paramI]
		numGates = len(param)
		for gateI in range(numGates):
			param[gateI][0] -= stepSize * grads[paramI][gateI][0]
			param[gateI][1] -= stepSize * grads[paramI][gateI][1]
		params[paramI] = param
	return params

# Compute accuracy
def accuracy(params, prevCell, prevHidden, curInput, targetVec, tokens, verbose):
	for paramI in range(len(params)):
		prevCell, prevHidden = jitLstmCell(params[paramI], prevCell, prevHidden, curInput[paramI])
	pred_token = jnp.argmax(prevHidden, axis = 0)
	target_token = jnp.argmax(targetVec, axis = 0)
	pred_char = vec2str(prevHidden, tokens)
	target_char = vec2str(targetVec, tokens)
	return pred_token == target_token, pred_char, target_char

# output vector to string transformation
def vec2str(vec, tokens):
	return tokens[jnp.argmax(vec, axis = 0)]

def dataPreProc(path):
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

# function optimizations
jitGradLstmSeqLoss = jit(grad(lstm_seq_loss, argnums = 0))

if (__name__ == '__main__'):
	## Data preprocessing
	## Preprocess the dataset
	# the path name
	path = './bbc-news-summary/BBC News Summary/News Articles/tech/'

	trainVec, testVec, tokens, seqMaxLen = dataPreProc(path)

	print('size of training data: ', len(trainVec))
	print('size of test data: ', len(testVec))
	print('maximum instance length: ', seqMaxLen)

	##################### Training of the LSTM model
	# model save path
	modelSavePath = './models/'

	# number of epoches to train for
	numEpoches = 1

	# custom the step size
	step_size = 0.1

	# custom the lstm size
	lstmSize = 20

	# size of the tokens
	tokensSize = len(tokens)

	# set the dimension of the parameters
	n = tokensSize
	m = tokensSize

	# the verbose
	verbose = False

	# initialize the first cell state
	cell_init = jnp.zeros([n,], dtype = float)

	# initialize the first hidden value
	hidden_init = jnp.zeros([m,], dtype = float)

	# initialize random parameters w and bias b
	params = []
	for tokenI in range(lstmSize):
	    param = []
	    for gateI in range(4):
	        w = random_params_by_size(n, m, jrandom.PRNGKey(0))
	        b = random_params_by_size(n, None, jrandom.PRNGKey(0))
	        param.append([w, b])
	    params.append(param)

	params = pickle.load(open(modelSavePath + "lstm_model.p", "rb"))

	# training epoches
	for epochI in range(numEpoches):
		## Train
		for instIndex in range(len(trainVec)):
			# get a training instance
			instance = trainVec[instIndex]
			
			# skip if instance is too short
			if (len(instance) <= lstmSize or len(instance) > lstmSize * 2): continue
			
			print('chosen training index: ', instIndex)
			while (1):
				print('on training index: ', instIndex)
				# initialize the cell and hidden
				prevCell = cell_init
				prevHidden = hidden_init

				# update the parameters
				for tokenI in range(0, len(instance) - lstmSize - 1):
					grads = jitGradLstmSeqLoss(params, prevCell, prevHidden,
						instance[tokenI : tokenI + lstmSize], instance[tokenI + lstmSize + 1])
					params = lstm_seq_update(params, grads, step_size)

					# get the next cell and hidden
					prevCell, prevHidden = lstm_seq(params, prevCell,
						prevHidden, instance[tokenI : tokenI + lstmSize])
					#print(tokenI, ' / ', len(instance) - lstmSize - 2)

				## See the performance
				totalLoss = 0
				totalAcc = 0

				# initialize the cell and hidden
				prevCell = cell_init
				prevHidden = hidden_init

				# initialize the predicted sequence and the target sequence
				pred_seq = ''
				target_seq = ''
				
				# test the training performance
				for tokenI in range(0, len(instance) - lstmSize - 1):
					totalLoss += lstm_seq_loss(params, prevCell, prevHidden,
						instance[tokenI : tokenI + lstmSize], instance[tokenI + lstmSize + 1])
					acc, pred_char, target_char = accuracy(params, prevCell, prevHidden,
						instance[tokenI : tokenI + lstmSize], instance[tokenI + lstmSize + 1],
						tokens, verbose)
					totalAcc += acc
					pred_seq += pred_char
					target_seq += target_char

					# get the next cell and hidden
					prevCell, prevHidden = lstm_seq(params, prevCell,
						prevHidden, instance[tokenI : tokenI + lstmSize])
				
				# compute the average accuracy
				avgAcc = float(totalAcc) / (tokenI + 1)
				
				# print the loss and accuracy
				print("loss: ", totalLoss)
				print("Accuracy: ", avgAcc)
				print(pred_seq)
				print(target_seq)
				print()
				
				# stop training on this instance if average accuracy is good enough
				if (avgAcc > 0.99): break

			pickle.dump(params, open(modelSavePath + "lstm_model.p", "wb"))
			break
			
			
			

