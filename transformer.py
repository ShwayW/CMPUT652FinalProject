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

## Implementation of components of the DTransformer
# the attention layer
# computes a single (masked) self- or cross- attention head
@jit
def attention(primary_seq, context_sequence, qkv, mask):
	# qkv[0] is W_query, b_query
	# qkv[1] is W_key, b_key
	# qkv[2] is W_value, b_value
	query = jnp.dot(qkv[0][0], primary_seq) + qkv[0][1]
	key = jnp.dot(qkv[1][0], context_sequence) + qkv[1][1]
	value = jnp.dot(qkv[2][0], context_sequence) + qkv[2][1]
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
def mhAttention(primary_seq, context_sequence, params, mask):
	# params is consist of:
	# H many triplets of qkv's and
	# W_o of size (d_out x H*d_mid), b_o of length d_out
	# in other words, H = len(params) - 1
	# multi-head attention loop
	cands = attention(primary_seq, context_sequence, params[0], mask)
	for h in range(1, len(params) - 1):
		cand = attention(primary_seq, context_sequence, params[h], mask)
		cands = jnp.concatentate((cands, cand), axis = 0)
	return jnp.dot(params[-1][0], cands) + params[-1][1]


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
def decoder_only_transformer(seq, pos_matrix, params, max_seq_len, L, H, d_enc, d_mlp):
	# seq is the input training sequence, is also contains the hard-coded token embedding
	# pos_matrix is the positional encoding matrix
	# params contains the weights for all the multi-head attention layers 
	seq_len = len(seq)
	for t in range(seq_len):
		seq[:, t] += pos_matrix[:, t]
		
	for l in range(L):
		for t in range(seq_len):
			seq[:, t] = rms_layer_norm(seq[:, t], )



	
if (__name__ == "__main__"):
	## Download the dataset
	############ DO NOT PUSH DATASET TO GITHUB #################
	#od.download("https://www.kaggle.com/datasets/pariza/bbc-news-summary")
	
	




