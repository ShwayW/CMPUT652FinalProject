# Shway Wang
# code modified from https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking

from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from numpy import array
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from encoder import Encoder
from decoder import Decoder


class TransformerModel(Model):
	def __init__(self, dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
		super(TransformerModel, self).__init__(**kwargs)
		
		# Set up the decoder
		self.decoder = Decoder(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)

		# Define the final dense layer
		self.model_last_layer = Dense(dec_vocab_size)
		
	def padding_mask(self, input):
		# Create mask which marks the zero padding values in the input by a 1
		mask = math.equal(input, 0)
		mask = cast(mask, float32)
		return mask[:, newaxis, newaxis, :]


	def lookahead_mask(self, shape):
		# Mask out future entries by marking them with a 1.0
		mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)
		return mask
		
		
	def call(self, decoder_input, training):
		# Create and combine padding and look-ahead masks to be fed into the decoder
		dec_in_padding_mask = self.padding_mask(decoder_input)
		dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
		dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask)
		
		# Feed the input into the decoder
		decoder_output = self.decoder(decoder_input, dec_in_lookahead_mask, training)
		
		# Pass the decoder output through a final dense layer
		return self.model_last_layer(decoder_output)
		
		
		
		
		
		
		
		
		
		
		
		
