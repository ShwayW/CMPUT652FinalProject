## The LSTM model
# Shway Wang
# Nov. 1, 2022

## Imports
# jax related
from jax.example_libraries import optimizers as jax_opt
from functools import partial
import jax.numpy as jnp
import jax.nn as jnn
from jax import grad, jit, vmap, lax, value_and_grad
from jax import random as jrandom
from jax.scipy.special import logsumexp

# others

import os
import pandas as pd
import opendatasets as od
import time
from random import randint
import pickle
from aux import *

## Download the dataset
############ DO NOT PUSH DATASET TO GITHUB #################
#od.download("https://www.kaggle.com/datasets/pariza/bbc-news-summary")

## Implementation of components of LSTM
# initialize the parameters randomly
def random_params_by_size(n, m, key, scale=1e-2):
    if (m is None):
        return scale * jrandom.normal(key, (n,))
    return scale * jrandom.normal(key, (n, m))

# an individual lstm cell
@jit
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
	return curCell, curHidden

# a sequence of lstm cells
@jit
def lstm_seq(params, prevCell, prevHidden, curInput):
	assert(len(params) == len(curInput))
	for inputI in range(len(curInput)):
		prevCell, prevHidden = lstm_cell(params[inputI], prevCell,
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
		prevCell, prevHidden = lstm_cell(params[inputI], prevCell,
			prevHidden, curInput[inputI])
	return jnp.mean(jnp.absolute(prevHidden - targetOutput))

# Compute accuracy
def accuracy(params, prevCell, prevHidden, curInput, targetVec, tokens, verbose):
	for paramI in range(len(params)):
		prevCell, prevHidden = lstm_cell(params[paramI], prevCell, prevHidden, curInput[paramI])
	pred_token = jnp.argmax(prevHidden, axis = 0)
	target_token = jnp.argmax(targetVec, axis = 0)
	pred_char = vec2str(prevHidden, tokens)
	target_char = vec2str(targetVec, tokens)
	return pred_token == target_token, pred_char, target_char

# function optimizations
jitValueGradLstmSeqLoss = jit(value_and_grad(lstm_seq_loss, argnums = 0))

def train_step(step_i, opt_state, prevCell, prevHidden, instance, lstmSize):
	# update the parameters
	totalLoss = 0
	for tokenI in range(len(instance) - lstmSize - 1):
		# Compute the loss and the gradients for the current parameters
		params = get_params(opt_state)
		loss, grads = jitValueGradLstmSeqLoss(params, prevCell, prevHidden,
			instance[tokenI : tokenI + lstmSize], instance[tokenI + lstmSize + 1])
		totalLoss += loss

		# update the parameters using adam and get the optimized state
		opt_state = jit(opt_update)(step_i, grads, opt_state)
		params = get_params(opt_state)

		# get the next cell and hidden
		prevCell, prevHidden = lstm_seq(params, prevCell, prevHidden,
				instance[tokenI : tokenI + lstmSize])

	return totalLoss, opt_state



if (__name__ == '__main__'):
	## Data preprocessing
	## Preprocess the dataset
	# the path name
	path = './bbc-news-summary/BBC News Summary/News Articles/tech/'

	trainVec, testVec, tokens, seqMaxLen = textDataPreProc(path)

	print('size of training data: ', len(trainVec))
	print('size of test data: ', len(testVec))
	print('maximum instance length: ', seqMaxLen)

	##################### Training of the LSTM model
	# model save path
	modelSavePath = './models/'

	# number of epoches to train for
	numEpoches = 1

	# custom the lstm size
	lstmSize = 200

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

	# use adam optimizer
	opt_init, opt_update, get_params = jax_opt.adam(0.001)
	opt_state = opt_init(params)


	#params = pickle.load(open(modelSavePath + "lstm_model.p", "rb"))

	# training epoches
	for epochI in range(numEpoches):
		## Train
		for instIndex in range(len(trainVec)):
			# get a training instance
			instance = trainVec[instIndex]
			
			# skip if instance is too short
			if (len(instance) <= lstmSize or len(instance) > lstmSize * 2): continue
			
			for step_i in range(1000):
				print('on training index: ', instIndex)
				# a training step
				totalLoss, opt_state = train_step(step_i, opt_state, cell_init, hidden_init, instance, lstmSize)

				# get the trained parameters from the optimized state
				params = get_params(opt_state)

				## See the performance
				totalAcc = 0

				# initialize the cell and hidden
				prevCell = cell_init
				prevHidden = hidden_init

				# initialize the predicted sequence and the target sequence
				pred_seq = ''
				target_seq = ''
				
				# test the training performance
				for tokenI in range(0, len(instance) - lstmSize - 1):
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
				if (avgAcc > 0.95): break

			pickle.dump(params, open(modelSavePath + "lstm_model.p", "wb"))
			
			
			

