# Shway Wang
# code modified from https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking

from pickle import load
from tensorflow import Module
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64, TensorArray, argmax, newaxis, transpose, data
from model import TransformerModel
from prepare_dataset import PrepareDataset
 
class Inference(Module):
	def __init__(self, inferencing_model, genLen, **kwargs):
		super(Inference, self).__init__(**kwargs)
		self.transformer = inferencing_model
		self.genLen = genLen

	def load_tokenizer(self, name):
		with open(name, 'rb') as handle:
			return load(handle)

	def convert_to_str(self, dec_tokenizer, decoder_output):
		output = transpose(decoder_output.stack())
		output = output.numpy()

		output_str = ''
		
		# Decode the predicted tokens into an output string
		for i in range(output.shape[0]):
			key = output[i]
			output_str += dec_tokenizer.index_word[key]
			
		return output_str
		
	def convert_display(self, output_str):
		display_str = ''
		for i in range(len(output_str)):
			display_str += output_str[i]
			if ((i + 1) % 32 == 0):
				display_str += '\n'
		return display_str

	def __call__(self, prompt, maxInputLen):
		# Load encoder and decoder tokenizers
		dec_tokenizer = self.load_tokenizer('levels_tokenizer.pkl')
		
		# Prepare the output <START> token by tokenizing, and converting to tensor
		output_start = dec_tokenizer.texts_to_sequences(prompt)
		output_start = convert_to_tensor(output_start[0], dtype = int64)

		# Prepare the output array of dynamic size
		decoder_output = TensorArray(dtype = int64, size = 0, dynamic_size = True)
		for i in range(len(output_start)):
			decoder_output = decoder_output.write(i, output_start[i])

		output_str = ''
		for genI in range(self.genLen):
			# Predict an output token
			prediction = self.transformer(decoder_output.stack()[newaxis, :], training = False)
			prediction = prediction[:, -1, :]

			# Select the prediction with the highest score
			predicted_id = argmax(prediction, axis = -1)[0]
			
			if (transpose(decoder_output.stack()).numpy().shape[0] >= maxInputLen):
				for i in range(maxInputLen - 1):
					decoder_output = decoder_output.write(i, decoder_output.read(i + 1))
				decoder_output = decoder_output.write(maxInputLen - 1, predicted_id)
			else:
				# Write the selected prediction to the output array at the next available index
				decoder_output = decoder_output.write(genI + len(output_start), predicted_id)
			
			print("pred index: ", genI)
			output_str += self.convert_to_str(dec_tokenizer, decoder_output)[-1]
			display_str = self.convert_display(output_str)
			print(display_str)
		return output_str

# the inference process:
if (__name__ == '__main__'):
	# the path to the trained model
	model_save_path = './models/transformer_speedrunner.h5'
	#model_save_path = './models/transformer_completionist.h5'

	# Define the model parameters
	h = 8  # Number of self-attention heads
	d_k = 64  # Dimensionality of the linearly projected queries and keys
	d_v = 64  # Dimensionality of the linearly projected values
	d_model = 512  # Dimensionality of model layers' outputs
	d_ff = 2048  # Dimensionality of the inner fully connected layer
	n = 6  # Number of layers in the encoder stack

	# batch size
	batch_size = 32
	
	# maximum input length
	maxInputLen = 256
	
	# the prompt
	prompt = '-------------xXXXXx-------------'
	prompt = '--------ox----XXXX----xo--------'
	
	# Desired generation length
	genLen = 1600
	
	# Use traned model
	useTrainedModel = True

	# Prepare the training and test splits of the dataset
	dataset = PrepareDataset()
	trainProc, train_orig, dec_seq_max_length, dec_vocab_size = dataset('./levels.pkl')
	
	print("Maximum sequence length: ", dec_seq_max_length)
	print("Vocabulary size: ", dec_vocab_size)
	
	# Prepare the dataset batches
	train_dataset = data.Dataset.from_tensor_slices(trainProc)
	train_dataset = train_dataset.batch(batch_size)


	# Create the inference model
	if (useTrainedModel):
		inferencing_model = TransformerModel(dec_vocab_size, dec_seq_max_length, h, d_k, d_v, d_model, d_ff, n, 0)
		
		# call the model before assigning the weights
		for step, train_batch in enumerate(train_dataset):
			# Define the encoder and decoder inputs, and the decoder output
			decoder_input = train_batch[:, :-1]
			inferencing_model(decoder_input, training = True)
			break
		
		# Load a pretrained model
		inferencing_model.load_weights(model_save_path)
		print("Restored from {}".format(model_save_path))
		inferencing_model.summary()
	else:		
		inferencing_model = TransformerModel(dec_vocab_size, dec_seq_max_length, h, d_k, d_v, d_model, d_ff, n, 0)
	
	inference = Inference(inferencing_model, genLen + len(prompt))
	
	# preprocess the prompt to char by char
	preced_prompt = []
	for i in range(len(prompt)):
		preced_prompt.append(prompt[i])
	
	pred = inference([preced_prompt], maxInputLen)
	
	print(pred)
	
	
	
	
	
	
	
