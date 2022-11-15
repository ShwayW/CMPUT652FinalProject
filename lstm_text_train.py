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

# others
import os
import pandas as pd
#import opendatasets as od
import time
from random import randint
import pickle
from aux import *

########## Adam optimizer declaration ##############
# use adam optimizer
opt_init, opt_update, get_params = jax_opt.adam(0.001)

# jit the optimizer functions
opt_init = jit(opt_init)
opt_update = jit(opt_update)
get_params = jit(get_params)

## Implementation of components of LSTM
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
	
# lstm improved
@jit
def lstm_seq_dense(params, prevCell, prevHidden, curInput):
	for i in range(len(params[1])):
		prevCell, prevHidden = lstm_seq(params[1][i], prevCell, prevHidden, curInput)
		prevHidden = jnp.dot(prevHidden, params[0][i])
	return prevCell, jnn.softmax(prevHidden)

## Implementation of the loss and update functions
# loss of lstm cells on a given input sequence predicting the next token
@jit
def lstm_seq_loss(params, prevCell, prevHidden, curInput, targetOutput):
	# curInput is a sequence
	prevCell, prevHidden = lstm_seq_dense(params, prevCell, prevHidden, curInput)
	return jnp.mean(jnp.absolute(prevHidden - targetOutput))

# function optimizations
jitValueGradLstmSeqLoss = jit(value_and_grad(lstm_seq_loss, argnums = 0))

# Compute accuracy
def accuracy(params, prevCell, prevHidden, curInput, targetVec, tokens):
	prevCell, prevHidden = lstm_seq_dense(params, prevCell, prevHidden, curInput)
	pred_tile_token = jnp.argmax(prevHidden, axis = 0)
	target_tile_token = jnp.argmax(targetVec, axis = 0)
	pred_char = vec2str(prevHidden, tokens)
	target_char = vec2str(targetVec, tokens)
	return pred_tile_token == target_tile_token, pred_char, target_char

# a single training step
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
		opt_state = opt_update(step_i, grads, opt_state)
		params = get_params(opt_state)

		# get the next cell and hidden
		prevCell, prevHidden = lstm_seq_dense(params, prevCell, prevHidden,
				instance[tokenI : tokenI + lstmSize])
	return totalLoss, opt_state


# To train the LSTM model
def train(numEpoches, trainVec, params, cell_init, hidden_init, lstmSize, tokens, modelSavePath):
	# get the initial opt_state
	opt_state = opt_init(params)
	
	## Train
	for instIndex in range(len(trainVec)):
		# training epoches
		for epochI in range(numEpoches):
			# get a training instance
			instance = trainVec[instIndex]
			
			# skip if instance is too short
			if (len(instance) <= lstmSize): continue
			
			print('on training index: ', instIndex, " epochI: ", epochI)
			# a training step
			totalLoss, opt_state = train_step(epochI, opt_state, cell_init, hidden_init, instance, lstmSize)

			# get the trained parameters from the optimized state
			params = get_params(opt_state)
			
			# save the result on the fly
			pickle.dump(params, open(modelSavePath, "wb"))

			## See the performance
			totalAcc = 0

			# initialize the cell and hidden
			prevCell = cell_init
			prevHidden = hidden_init

			# initialize the predicted sequence and the target sequence
			pred_tiles = ''
			target_tiles = ''
			
			# test the training performance
			for tokenI in range(0, len(instance) - lstmSize - 1):
				acc, pred, target = accuracy(params, prevCell, prevHidden,
					instance[tokenI : tokenI + lstmSize], instance[tokenI + lstmSize + 1],
					tokens)
				totalAcc += acc
				pred_tiles += pred
				target_tiles += target

				# get the next cell and hidden
				prevCell, prevHidden = lstm_seq_dense(params, prevCell,
					prevHidden, instance[tokenI : tokenI + lstmSize])
			
			# compute the average accuracy
			avgAcc = float(totalAcc) / (tokenI + 1)
			
			# print the loss and accuracy
			print("loss: ", totalLoss)
			print("Accuracy: ", avgAcc)
			print(pred_tiles)
			print(target_tiles)
			print()
			if (avgAcc > 0.95): break
	return params

# initialize the parameters for the LSTM model
def init_lstm_params(lstmSize, n, m):
	params = []
	for tokenI in range(lstmSize):
	    param = []
	    for gateI in range(4):
	        w = random_params_by_size(n, m)
	        b = random_params_by_size(n, None)
	        param.append([w, b])
	    params.append(param)
	return params


if (__name__ == '__main__'):
	## Download the dataset
	############ DO NOT PUSH DATASET TO GITHUB #################
	#od.download("https://www.kaggle.com/datasets/pariza/bbc-news-summary")

	## Data preprocessing
	## Preprocess the dataset
	# the path name
	path = './bbc-news-summary/BBC News Summary/News Articles/tech/'

	trainVec, testVec, tokens, seqMaxLen = textDataPreProc(path)

	print('size of training data: ', len(trainVec))
	print('size of test data: ', len(testVec))
	print('size of tokens: ', len(tokens))
	print('maximum instance length: ', seqMaxLen)

	##################### Training of the LSTM model
	# model save path
	modelSavePath = './models/lstm_model_text.pickle'

	# flag for training again
	trainAgain = True

	# number of epoches to train for
	numEpoches = 200

	# custom the lstm size
	lstmSize = 20
	denseSize = 3

	# size of the tokens
	tokensSize = len(tokens)

	# set the dimension of the parameters
	n = tokensSize
	m = tokensSize

	# initialize the first cell state
	cell_init = jnp.zeros([n,], dtype = float)

	# initialize the first hidden value
	hidden_init = jnp.zeros([m,], dtype = float)

	# initialize random parameters w and bias b
	if (trainAgain):
		dense_params = []
		lstm_params = []
		for i in range(denseSize):
			dense_params.append(random_params_by_size(n, m))
			lstm_params.append(init_lstm_params(lstmSize, n, m))
		params = [dense_params, lstm_params]
	else:
		params = pickle.load(open(modelSavePath, "rb"))

	# training start here
	params = train(numEpoches, trainVec, params, cell_init, hidden_init, lstmSize, tokens, modelSavePath)
	
	# save results here
	pickle.dump(params, open(modelSavePath, "wb"))
	
	
	
