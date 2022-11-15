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
	# qkv[0] is W_query, b_query
	# qkv[1] is W_key, b_key
	# qkv[2] is W_value, b_value
	query = jnp.dot(qkv[0][0], primary_seq) + qkv[0][1]
	key = jnp.dot(qkv[1][0], primary_seq) + qkv[1][1]
	value = jnp.dot(qkv[2][0], primary_seq) + qkv[2][1]
	score = jnp.dot(jnp.transpose(key), query)
	
	# apply the mask
	for i in range(len(mask)):
		for j in range(len(mask[i])):
			if (not mask[i, j]):
				score[i, j] = -jnp.Inf
	
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
	cands = attention(primary_seq, W_l[0][0], mask)
	for h in range(1, len(W_l[0])):
		cand = attention(primary_seq, W_l[0][h], mask)
		cands = jnp.concatentate((cands, cand), axis = 0)
	return jnp.dot(W_l[1][0], cands) + W_l[1][1]


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


# initialize the parameters for the transformer model
@jit
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


# the decoder-only transformer
@jit
def decoder_only_transformer(seq, params, vocabSize, seqMaxLen, L, H, d_enc, d_mlp, d_attn, d_x, d_z, d_mid, d_out):
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
	# unpack params
	[W_enc, W_pos, W_mhattn, scale_l_fin, W_u] = params
	
	# initialize the mask
	# mask[t, t_prime] = [t <= t_prime] is equivalent to below:
	mask = jnp.ones([seq_len, seq_len])
	for t in range(1, seq_len):
		mask = mask.at[t, :t].set(0)
	
	############### Transformer implementation #############
	# get the length of the sequence
	seq_len = len(seq)
	
	# convert the sequence into token + positional embeddings
	X = W_enc[:, seq[0]] + W_pos[:, 0]
	for t in range(1, seq_len):
		X = jnp.concatenate(X, W_enc[:, seq[t]] + W_pos[:, t], axis = 1)
	
	# for each layer of the transformer, compute on the encoded sequence X
	for l in range(L):
		# unpack W_mhattn[l]
		[W_l, scale_l_fst, scale_l_scd, W_l_mlp_fst, b_l_mlp_fst, W_l_mlp_scd, b_l_mlp_scd] = W_mhattn[l]
		
		# apply the first layer norm to X
		X_cp = rms_layer_norm(X[:, 0], scale_l_fst)
		for t in range(1, seq_len):
			X_cp = jnp.concatenate(X_cp, rms_layer_norm(X[:, t], scale_l_fst), axis = 1)
		
		# apply multi-head attention
		X = X + mhAttention(X_cp, W_l, mask)
		
		# apply the second layer norm to X
		X_cp = rms_layer_norm(X[:, 0], scale_l_scd)
		for t in range(1, seq_len):
			X_cp = jnp.concatenate(X_cp, rms_layer_norm(X[:, t], scale_l_scd), axis = 1)
			
		# apply GELU and MLP's to X_cp
		Gelu = jnn.gelu(jnp.dot(W_l_mlp_fst, X_cp) + b_l_mlp_fst)
		X = X + jnp.dot(W_l_mlp_scd, Gelu) + b_l_mlp_scd
		
	# apply the final layer norm to X
	X = rms_layer_norm(X[:, 0], scale_l_fin)
	for t in range(1, seq_len):
		X = jnp.concatenate(X, rms_layer_norm(X[:, t], scale_l_fin), axis = 1)
		
	# apply unembedding and softmax
	return jnn.softmax(jnp.dot(W_u, X))


# Training of the transformer
def train(dataset, params):
	
	
	
	return params

	
if (__name__ == "__main__"):
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
	print('maximum instance length: ', seqMaxLen)

	##################### Training of the LSTM model #################
	# model save path
	modelSavePath = './models/'

	# number of epoches to train for
	numEpoches = 1

	# custom the lstm size
	lstmSize = 200

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
	params = init_params(vocabSize, seqMaxLen, L, H, d_enc, d_mlp, d_attn, d_x, d_z, d_mid, d_out)
	
	




