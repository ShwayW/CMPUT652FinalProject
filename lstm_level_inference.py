## The LSTM model inference
# Shway Wang
# Nov. 1, 2022

## Imports
# jax related
from jax.example_libraries import optimizers as jax_opt
from functools import partial
import jax.numpy as jnp
import jax.nn as jnn
from jax import grad, jit, vmap, lax, value_and_grad, random

# others
import os
import pandas as pd
#import opendatasets as od
import time
from random import randint, choices
import pickle
from aux import *

## Implementation of components of LSTM
# the Drop out layer
@jit
def dropout(seed, params):
	# Reminder: [W_i, dense_params, lstm_params] = params
	rate = 0.1
	for j in range(len(params[2])):
		key = random.PRNGKey(seed + j)
		param_idxs = choices(range(len(params[2][j])), k = int(rate * len(params[2][j])))
		for i in param_idxs:
			param = params[2][j][i]
			for gateI in range(4):
				isKeep = random.bernoulli(key, rate, param[gateI][0].shape)
				param[gateI][0] = jnp.where(isKeep, param[gateI][0] / rate, 0)
				isKeep = random.bernoulli(key, rate, param[gateI][1].shape)
				param[gateI][1] = jnp.where(isKeep, param[gateI][1] / rate, 0)
	return params


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
	hiddenInput = []
	for inputI in range(len(curInput)):
		prevCell, prevHidden = lstm_cell(params[inputI], prevCell,
			prevHidden, curInput[inputI])
		hiddenInput.append(prevHidden)
	return prevCell, prevHidden, hiddenInput


# lstm improved with dense layers
@jit
def lstm_seq_dense(params, initCell, initHidden, curInput):
	# Reminder: [W_i, dense_params, lstm_params] = params
	# Apply the first dense layer W_i
	hiddenInput = jnp.dot(params[0], jnp.asarray(curInput))
	
	# Apply LSTM layers
	for i in range(len(params[2])):
		# Reminder: lstm_params = params[2]
		prevCell, prevHidden, hiddenInput = lstm_seq(params[2][i], initCell, initHidden, hiddenInput)
		
		# Reminder: W = params[1][i]
		hiddenInput = jnp.dot(params[1][i], jnp.asarray(hiddenInput))
	
	# apply drop out
	#params = dropout(0, params)

	return jnn.gelu(jnp.sum(hiddenInput, axis = 0))
	

# To train the LSTM model
def inference(randInput, params, cell_init, hidden_init, lstmSize, numOutputs, tokens):
	## See the performance
	totalAcc = 0

	# initialize the cell and hidden
	prevCell = cell_init
	prevHidden = hidden_init

	# initialize the predicted sequence and the target sequence
	pred_tiles = ''
	pred_path = ''
	
	# test the training performance
	for tokenI in range(0, numOutputs):
		prevHidden = lstm_seq_dense(params, prevCell, prevHidden, randInput)
		pred = vec2tile(prevHidden, tokens)
		
		ai = np.argmax(prevHidden[:-2], axis = 0)
		bi = np.argmax(prevHidden[-2:], axis = 0)
		
		prevHidden = prevHidden.at[:].set(0)
		prevHidden = prevHidden.at[ai].set(1)
		prevHidden = prevHidden.at[len(prevHidden[:-2]) + bi].set(1)
		
		pred_tiles += pred[0]
		pred_path += pred[1]

		# get the next input
		
		prevHidden = prevHidden.at[:].set(0)
		
		ai = np.random.choice(len(prevHidden[:-2]), 1)
		bi = np.random.choice(2, 1)
		
		prevHidden = prevHidden.at[ai].set(1)
		prevHidden = prevHidden.at[len(prevHidden[:-2]) + bi].set(1)
		
		randInput = randInput[1:]
		randInput = jnp.row_stack((randInput, prevHidden))
	
	# print the output
	print(pred_tiles)
	print(pred_path)
	print()
	return [pred_tiles, pred_path]

if (__name__ == "__main__"):
	trainVec, tokens, seqMaxLen = processPathAndLevel()
	print(len(trainVec[0]))
	# get the randInput
	randInput = trainVec[0][:100]
	for i in range(len(randInput)):
		randInput[i][:] = 0
		
		ai = np.random.choice(len(randInput[i][:-2]), 1)
		bi = np.random.choice(2, 1)
		
		randInput[i][ai] = 1
		randInput[i][len(randInput[i][:-2]) + bi] = 1
	
	##################### Training of the LSTM model
	# model load path
	modelLoadPath = './models/lstm_model_smb.pickle'

	# output save path
	outputSavePath = './output/output_spd_2.txt'

	# number of tokens as inputs
	numOutputs = 3232

	# custom the lstm size
	lstmSize = 100

	# size of n
	n = len(tokens)

	# initialize the first cell state
	cell_init = jnp.zeros([n,], dtype = float)

	# initialize the first hidden value
	hidden_init = jnp.zeros([n,], dtype = float)

	# initialize random parameters w and bias b
	params = pickle.load(open(modelLoadPath, "rb"))
	
	# training start here
	output = inference(randInput, params, cell_init, hidden_init, lstmSize, numOutputs, tokens)
	
	with open(outputSavePath, 'w') as f:
		f.write(output[0])
		f.write('\n')
	with open(outputSavePath, 'a') as f:
		f.write(output[1])
	
	
	
