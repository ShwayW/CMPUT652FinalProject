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

## Implementation of the loss and update functions
# loss of a single lstm cell
def lstm_cell_loss(param, prevCell, prevHidden, curInput, targetOutput):
	prevCell, prevHidden = lstm_cell(param, prevCell, prevHidden, curInput)
	return jnp.mean(jnp.absolute(prevHidden - targetOutput))

# lstm cell update
def lstm_cell_update(param, grads, stepSize):
    numGates = len(param)
    for gateI in range(numGates):
        param[gateI][0] -= stepSize * grads[gateI][0]
        param[gateI][1] -= stepSize * grads[gateI][1]
    return param

# Compute accuracy
def accuracy(param, prevCell, prevHidden, curToken, targetVec, tokens, verbose):
    prevCell, prevHidden = lstm_cell(param, prevCell, prevHidden, curToken)
    pred_token = jnp.argmax(prevHidden, axis = 0)
    target_token = jnp.argmax(targetVec, axis = 0)
    if (verbose):
	    pred_char = vec2str(prevHidden, tokens)
	    target_char = vec2str(targetVec, tokens)
	    print(pred_char, target_char)
    return pred_token == target_token

# output vector to string transformation
def vec2str(vec, tokens):
	return tokens[jnp.argmax(vec, axis = 0)]

# pre-compile costly functions
jitUpdate = jit(lstm_cell_update)
jitLoss = jit(lstm_cell_loss)
jitAccuracy = jit(accuracy)

if __name__ == '__main__':
	## Data preprocessing
	## Preprocess the dataset
	# the path name
	path = 'bbc-news-summary/BBC News Summary/News Articles/tech/{}.txt'

	lines = []

	for i in range(1, 402):
	    if (i < 10):
	        index = "00{}".format(i)
	    elif (i < 100):
	        index = "0{}".format(i)
	    else:
	        index = "{}".format(i)
	    with open(path.format(index), 'r') as f:
	        lines.extend(f.readlines())

	# Remove new lines and empty lines
	lines = [line for line in lines if line != '\n' or '']

	# Strip all lines
	lines = [line.strip() for line in lines]

	# remove lines that are less than 10 tokens
	lines = [line for line in lines if len(line) > 10]

	# get the unique tokens
	tokens = getTokens(lines)

	# split the lines to training set and test set
	splitInd = floor(0.8 * len(lines))
	train = lines[:splitInd + 1]
	test = lines[splitInd + 1:]

	# convert the training set and test set into one hot vectors
	trainVec = toOneHot(train, tokens)
	testVec = toOneHot(test, tokens)

	print(len(trainVec))
	print(len(train))

	print(len(testVec))
	print(len(test))

	##################### Training of the LSTM model
	# number of epoches to train for
	numEpoches = 100

	# custom the step size
	step_size = 0.1

	# number of modules in the LSTM model
	lstmSize = 10

	# number of instances to train
	numInst = 2

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

	# the gradient function of the loss
	gradLoss = grad(jitLoss, argnums = 0)

	for instI in range(numInst):
		# get the training instance
		instance = trainVec[instI]

		# initialize random parameters w and bias b, n + 1 accounts for the bias
		params = []
		for tokenI in range(len(instance)):
		    param = []
		    for gateI in range(4):
		        w = random_params_by_size(n, m, jrandom.PRNGKey(0))
		        b = random_params_by_size(n, None, jrandom.PRNGKey(0))
		        param.append([w, b])
		    params.append(param)

		# training epoches
		for epochI in range(numEpoches):
			## Train
			# initialize the cell and hidden
			prevCell = cell_init
			prevHidden = hidden_init

			# update the parameters
			for tokenI in range(1, len(instance)):
				grads = gradLoss(params[tokenI], prevCell, prevHidden,
					instance[tokenI - 1], instance[tokenI])
				params[tokenI] = jitUpdate(params[tokenI], grads, step_size)

		        # get the next cell and hidden
				prevCell, prevHidden = lstm_cell(params[tokenI], prevCell,
		        	prevHidden, instance[tokenI - 1])

			## Test
			totalLoss = 0
			totalAcc = 0

			# initialize the cell and hidden
			prevCell = cell_init
			prevHidden = hidden_init

			# test the training performance
			for tokenI in range(1, len(instance)):
				totalLoss += lstm_cell_loss(params[tokenI], prevCell, prevHidden,
					instance[tokenI - 1], instance[tokenI])
				totalAcc += accuracy(params[tokenI], prevCell, prevHidden,
					instance[tokenI - 1], instance[tokenI], tokens, verbose)

				# get the next cell and hidden
				prevCell, prevHidden = lstm_cell(params[tokenI], prevCell,
					prevHidden, instance[tokenI - 1])

			print("loss: ", totalLoss)
			print("Accuracy: ", float(totalAcc) / (len(instance) - 1))
