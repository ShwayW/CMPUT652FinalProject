# Shway Wang
# code modified from https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking

from tensorflow.keras.layers import Layer, Dropout, LayerNormalization, Dense
from tensorflow.keras.activations import gelu
from multihead_attention import MHAttention
from positional_encoding import PositionEmbeddingFixedWeights

# Implementing the Add & Norm Layer
class AddNormalization(Layer):
	def __init__(self, **kwargs):
		super(AddNormalization, self).__init__(**kwargs)
		self.layer_norm = LayerNormalization()  # Layer normalization layer

	def call(self, x, sublayer_x):
		# The sublayer input and output need to be of the same shape to be summed
		add = x + sublayer_x

		# Apply layer normalization to the sum
		return self.layer_norm(add)

# Implementing the Feed-Forward Layer
class FeedForward(Layer):
	def __init__(self, d_ff, d_model, **kwargs):
		super(FeedForward, self).__init__(**kwargs)
		self.fully_connected1 = Dense(d_ff)  # First fully connected layer
		self.fully_connected2 = Dense(d_model)  # Second fully connected layer

	def call(self, x):
		# The input is passed into the two fully-connected layers, with a GeLU in between
		x_fc1 = self.fully_connected1(x)
		return self.fully_connected2(gelu(x_fc1))
		#return self.fully_connected2(x_fc1)

# Implementing the Decoder Layer
class DecoderLayer(Layer):
	def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
		super(DecoderLayer, self).__init__(**kwargs)
		self.norm1 = LayerNormalization()
		self.multihead_attention = MHAttention(h, d_k, d_v, d_model)
		self.dropout1 = Dropout(rate)
		self.add_norm1 = AddNormalization()
		self.feed_forward = FeedForward(d_ff, d_model)
		self.dropout2 = Dropout(rate)
		

	def call(self, x, lookahead_mask, training):
		# Layer Normalization
		norm_output1 = self.norm1(x)
		# Expected output shape = (batch_size, sequence_length, d_model)
	
		# Multi-head attention layer
		multihead_output1 = self.multihead_attention(norm_output1, norm_output1, lookahead_mask)
		# Expected output shape = (batch_size, sequence_length, d_model)

		# Add in a dropout layer
		multihead_output1 = self.dropout1(multihead_output1, training = training)

		# Followed by an Add & Norm layer
		addnorm_output1 = self.add_norm1(x, multihead_output1)
		# Expected output shape = (batch_size, sequence_length, d_model)

		# Followed by a fully connected layer
		feedforward_output = self.feed_forward(addnorm_output1)
		# Expected output shape = (batch_size, sequence_length, d_model)

		# Add in another dropout layer
		feedforward_output = self.dropout2(feedforward_output, training = training)

		# Return X + feedforward_output
		return x + feedforward_output
 
# Implementing the Decoder
class Decoder(Layer):
	def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
		super(Decoder, self).__init__(**kwargs)
		self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
		self.dropout = Dropout(rate)
		self.decoder_layer = [DecoderLayer(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]
		self.norm_fin = LayerNormalization()

	def call(self, output_target, lookahead_mask, training):
		# Generate the positional encoding
		pos_encoding_output = self.pos_encoding(output_target)
		# Expected output shape = (number of sentences, sequence_length, d_model)

		# Add in a dropout layer
		x = self.dropout(pos_encoding_output, training = training)

		# Pass on the positional encoded values to each encoder layer
		for i, layer in enumerate(self.decoder_layer):
			x = layer(x, lookahead_mask, training)
		
		# Add a final normalization layer
		return self.norm_fin(x)
		
		
		
