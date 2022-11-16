## The Decoder-only Transformer model, since we want to do sequence modeling
# implementation details refer to https://arxiv.org/pdf/2207.09238.pdf
# Shway Wang
# Nov. 10, 2022

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
import opendatasets as od
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

## Implementation of components of the DTransformer
# token embedding
@jit
def token_embedding(token_id, W_enc):
	return W_enc[:, token_id]

# positional embedding
@jit
def pos_embedding(pos, W_p):
	return W_p[:, pos]

# the attention layer
# computes a single (masked) self- or cross- attention head
@jit
def attention(primary_seq, qkv, mask):
	# qkv[0:2] is W_query, b_query
	# qkv[2:4] is W_key, b_key
	# qkv[4:6] is W_value, b_value
	query = jnp.dot(qkv[0], primary_seq) + jnp.sum(qkv[1])
	key = jnp.dot(qkv[2], primary_seq) + jnp.sum(qkv[3])
	value = jnp.dot(qkv[4], primary_seq) + jnp.sum(qkv[5])
	score = jnp.dot(jnp.transpose(key), query)
	
	# apply the mask
	score = jnp.multiply(score, mask)
	
	# return the candidate value
	return jnp.dot(value, jnn.softmax(score/jnp.sqrt(len(key))))


# the multi-head attention layer
# computes multi-head (masked) self- or cross-attention layer
@jit
def mhAttention(primary_seq, W_l, mask):
	# W_l is consist of:
	# H many triplets of qkv's and
	# W_o of size (d_out x H*d_mid), b_o of length d_out
	# In other words:
	# W_l = [
	#		[[W_q, b_q, W_k, b_k, W_v, b_v] for h in range(H)]
	#		[W_o, b_o]
	#		]
	
	# Thus, H = len(W_l[0])
	# multi-head attention loop
	cands = []
	for h in range(0, len(W_l[0])):
		cand = attention(primary_seq, W_l[0][h], mask)
		cands.extend(cand)
	cands = jnp.asarray(cands)
	return jnp.dot(W_l[1][0], cands) + jnp.sum(W_l[1][1])


# root mean square layer normalization: mean and offset are set to 0
# normalizes layer activations
@jit
def rms_layer_norm(nn_activations, ele_wise_scale):
	return jnp.multiply(nn_activations/jnp.sqrt(jnp.var(nn_activations)), ele_wise_scale)


# unembedding converts a vector representation of a token and its context into
# into a distribution over the vocabulary elements
@jit
def unembedding(token_encoding, unemb_matrix):
	return jnn.softmax(jnp.dot(unemb_matrix, token_encoding))


# the decoder-only transformer
@jit
def decoder_only_transformer(seq, params):
	# seq is the input training sequence, is also contains the hard-coded token embedding
	# pos_matrix is the positional encoding matrix
	# params contains the weights for all the multi-head attention layers 
	############### Content of params ######################
	# params = [W_enc, W_pos, W_mhattn, scale_l_fin, W_u]
	# W_mhattn = [W_l, scale_l_fst, scale_l_scd, W_l_mlp_fst, b_l_mlp_fst, W_l_mlp_scd, b_l_mlp_scd for l in range(L)]
	# W_l = [
	#		[[W_q, b_q, W_k, b_k, W_v, b_v] for h in range(H)]
	#		[W_o, b_o]
	#		]
	
	############### Preliminaries ##########################
	# get the length of the sequence
	seq_len = len(seq)
	
	# unpack params
	[W_enc, W_pos, W_mhattn, scale_l_fin, W_u] = params
	
	# initialize the mask
	# mask[t, t_prime] = [t <= t_prime] is equivalent to below:
	mask = jnp.ones([seq_len, seq_len])
	for t in range(1, seq_len):
		mask = mask.at[t, :t].set(-jnp.inf)
	
	############### Transformer implementation #############
	# convert the sequence into token + positional embeddings
	X = []
	for t in range(0, seq_len):
		X.append(W_enc[:, seq[t]] + W_pos[:, t])
	X = jnp.asarray(X).T
	
	# for each layer of the transformer, compute on the encoded sequence X
	for l in range(len(W_mhattn)):
		# unpack W_mhattn[l]
		[W_l, scale_l_fst, scale_l_scd, W_l_mlp_fst, b_l_mlp_fst, W_l_mlp_scd, b_l_mlp_scd] = W_mhattn[l]
		
		# apply the first layer norm to X
		X_cp = []
		for t in range(0, seq_len):
			X_cp.append(rms_layer_norm(X[:, t], scale_l_fst))
		X_cp = jnp.asarray(X_cp).T
		
		# apply multi-head attention
		X = X + mhAttention(X_cp, W_l, mask)
		
		# apply the second layer norm to X
		X_cp = []
		for t in range(0, seq_len):
			X_cp.append(rms_layer_norm(X[:, t], scale_l_scd))
		X_cp = jnp.asarray(X_cp).T
			
		# apply GELU and MLP's to X_cp
		Gelu = jnn.gelu(jnp.dot(W_l_mlp_fst, X_cp) + jnp.sum(b_l_mlp_fst))
		X += jnp.dot(W_l_mlp_scd, Gelu) + jnp.sum(b_l_mlp_scd)
		
	# apply the final layer norm to X
	X_res = []
	for t in range(0, seq_len):
		X_res.append(rms_layer_norm(X[:, t], scale_l_fin))
	X_res = jnp.asarray(X_res).T
		
	# apply unembedding and softmax
	return jnn.softmax(jnp.dot(W_u, X_res))



# initialize the parameters for the transformer model
def init_params(vocabSize, seqMaxLen, L, H, d_enc, d_mlp, d_attn, d_x, d_z, d_mid, d_out):
	# token encoding and positional encoding
	W_enc = random_params_by_size(d_enc, vocabSize)
	W_pos = random_params_by_size(d_enc, seqMaxLen)
	
	# attention related parameters
	W_mhattn = []
	for l in range(L):
		# parameters for the multi-head attention on level l
		W_l = []
		W_l_attn = []
		for h in range(H):
			# each head in the multi-head attention layer
			W_q = random_params_by_size(d_attn, d_x)
			b_q = random_params_by_size(d_attn, None)
			W_k = random_params_by_size(d_attn, d_z)
			b_k = random_params_by_size(d_attn, None)
			W_v = random_params_by_size(d_mid, d_z)
			b_v = random_params_by_size(d_mid, None)
			W_l_attn.append([W_q, b_q, W_k, b_k, W_v, b_v])
		# the aggregate output matrix and bias
		W_o = random_params_by_size(d_out, H * d_mid)
		b_o = random_params_by_size(d_out, None)
		W_l.extend([W_l_attn, [W_o, b_o]])
		
		# two sets of layer norm element-wise scales
		scale_l_fst = random_params_by_size(d_enc, None)
		scale_l_scd = random_params_by_size(d_enc, None)
		
		# two sets of Multi-Level Perceptrons (MLP) parameters
		W_l_mlp_fst = random_params_by_size(d_mlp, d_enc)
		b_l_mlp_fst = random_params_by_size(d_mlp, None)
		W_l_mlp_scd = random_params_by_size(d_enc, d_mlp)
		b_l_mlp_scd = random_params_by_size(d_enc, None)
		
		# append the parameters of the current level to W_mhattn
		W_mhattn.append([W_l, scale_l_fst, scale_l_scd, W_l_mlp_fst, b_l_mlp_fst, W_l_mlp_scd, b_l_mlp_scd])
		
	# final layer norm parameters
	scale_l_fin = random_params_by_size(d_enc, None)
	
	# unembedding matrix
	W_u = random_params_by_size(vocabSize, d_enc)
	
	# return the parameters as a list
	return [W_enc, W_pos, W_mhattn, scale_l_fin, W_u]

# the Loss Function
@jit
def transformerLoss(params, seq):
	# compute the distribution from the transformer
	distrib = decoder_only_transformer(seq, params)
	
	# compute the loss
	loss = 0
	for t in range(len(seq) - 1):
		loss -= jnp.log(distrib[seq[t + 1], t])
	
	# output the loss
	return loss

# function optimizations for gradient of loss function
jitValueGradLoss = jit(value_and_grad(transformerLoss, argnums = 0))


# Training of the transformer
def train(train, params, numEpoches, modelSavePath, checkPoint):
	# get the initial opt_state
	opt_state = opt_init(params)
	
	# step_i for the adam optimizer
	step_i = 0
	
	# training loop
	for i in range(numEpoches):
		for n in range(len(train)):
			# get the params from the input opt_state
			params = get_params(opt_state)

			print(train[n])

			# the loss and the grads
			#loss = transformerLoss(params, seq)
			loss, grads = jitValueGradLoss(params, train[n])
			
			# update the parameters using adam and get the optimized state
			opt_state = opt_update(step_i, grads, opt_state)
			
			if (checkPoint):
				# save results here
				params = get_params(opt_state)
				pickle.dump(params, open(modelSavePath, "wb"))
			
			print("epochI: ", i, "instanceI: ", n, "step_i: ", step_i, "loss: ", loss)
			
			# increment the step_i
			step_i += 1
			
		# display accuracy
		
	params = get_params(opt_state)
	return params

	
if (__name__ == "__main__"):
	## Download the dataset
	############ DO NOT PUSH DATASET TO GITHUB #################
	#od.download("https://www.kaggle.com/datasets/pariza/bbc-news-summary")
	
	## Data preprocessing
	## Preprocess the dataset
	# the path name
	path = './bbc-news-summary/BBC News Summary/News Articles/tech/'

	trainData, testData, tokens, seqMaxLen = textDataPreProcTransformer(path)

	print('size of training data: ', len(trainData))
	print('size of test data: ', len(testData))
	print('maximum instance length: ', seqMaxLen)

	##################### Training of the LSTM model #################
	# model save path
	modelSavePath = './models/transformer_text.pickle'
	
	# flag for training again
	trainAgain = True
	
	# if we want check points
	checkPoint = True

	# number of epoches to train for
	numEpoches = 2

	# size of the tokens
	vocabSize = len(tokens)
	
	################## initialize the parameters #####################
	# number of levels in the decoder-only transformer
	L = 3
	
	# number of heads in the multi-head attention layer
	H = 3
	
	# encoder dimension
	d_enc = 20
	
	# MLP dimension
	d_mlp = 20
	
	# attention dimension
	d_attn = 20
	
	# primary sequence dimension
	d_x = 20
	
	# context sequence dimension
	d_z = 20
	
	# middle dimension
	d_mid = 20
	
	# output dimension
	d_out = 20
	
	# initialize the parameters
	if (trainAgain):
		params = init_params(vocabSize, seqMaxLen, L, H, d_enc, d_mlp, d_attn, d_x, d_z, d_mid, d_out)
	else:
		params = pickle.load(open(modelSavePath, "rb"))
	
	#### Training ####
	params = train(trainData, params, numEpoches, modelSavePath, checkPoint)
	
	# save results here
	pickle.dump(params, open(modelSavePath, "wb"))




