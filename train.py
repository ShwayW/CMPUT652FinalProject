# Shway Wang
# code modified from https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.metrics import Mean
from tensorflow import math, data, train, reduce_sum, equal, argmax, GradientTape, TensorSpec, function, int64, cast, float32
from keras.losses import sparse_categorical_crossentropy
from keras.models import load_model
from model import TransformerModel
from prepare_dataset import PrepareDataset
from time import time

# Defining the loss function
def loss_fcn(target, prediction):
	# Create mask so that the zero padding values are not included in the computation of loss
	padding_mask = math.logical_not(equal(target, 0))
	padding_mask = cast(padding_mask, float32)

	# Compute a sparse categorical cross-entropy loss on the unmasked values
	loss = sparse_categorical_crossentropy(target, prediction, from_logits = True) * padding_mask

	# Compute the mean loss over the unmasked values
	return reduce_sum(loss) / reduce_sum(padding_mask)
 
 
# Defining the accuracy function
def accuracy_fcn(target, prediction):
	# Create mask so that the zero padding values are not included in the computation of accuracy
	padding_mask = math.logical_not(equal(target, 0))

	# Find equal prediction and target values, and apply the padding mask
	accuracy = equal(target, argmax(prediction, axis = 2))
	accuracy = math.logical_and(padding_mask, accuracy)

	# Cast the True/False values to 32-bit-precision floating-point numbers
	padding_mask = cast(padding_mask, float32)
	accuracy = cast(accuracy, float32)

	# Compute the mean accuracy over the unmasked values
	return reduce_sum(accuracy) / reduce_sum(padding_mask)

# Speeding up the training process
@function
def train_step(training_model, decoder_input, decoder_output):
	with GradientTape() as tape:
		# Run the forward pass of the model to generate a prediction
		prediction = training_model(decoder_input, training = True)
		
		# Compute the training loss
		loss = loss_fcn(decoder_output, prediction)
		
		# Compute the training accuracy
		accuracy = accuracy_fcn(decoder_output, prediction)

	# Retrieve gradients of the trainable variables with respect to the training loss
	gradients = tape.gradient(loss, training_model.trainable_weights)

	# Update the values of the trainable variables by gradient descent
	optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))

	train_loss(loss)
	train_accuracy(accuracy)
	
	

# Implementing a learning rate scheduler
class LRScheduler(LearningRateSchedule):
	def __init__(self, d_model, warmup_steps=4000, **kwargs):
		super(LRScheduler, self).__init__(**kwargs)
		self.d_model = cast(d_model, float32)
		self.warmup_steps = warmup_steps

	def __call__(self, step_num):
		# Linearly increasing the learning rate for the first warmup_steps, and decreasing it thereafter
		arg1 = step_num ** -0.5
		arg2 = step_num * (self.warmup_steps ** -1.5)
		return (self.d_model ** -0.5) * math.minimum(arg1, arg2)



if (__name__ == '__main__'):
	level_style = 'speedrunner'
	#level_style = 'completionist'

	if (level_style == 'speedrunner'):
		datasetFilename = './speedrunner_levels.pkl'
		model_save_path = './models/transformer_speedrunner.h5'
	elif (level_style == 'completionist'):
		datasetFilename = './completionist_levels.pkl'
		model_save_path = './models/transformer_completionist.h5'
	
	# If we want to train again
	trainAgain = False

	# Define the model parameters
	h = 8  # Number of self-attention heads
	d_k = 64  # Dimensionality of the linearly projected queries and keys
	d_v = 64  # Dimensionality of the linearly projected values
	d_model = 512  # Dimensionality of model layers' outputs
	d_ff = 2048  # Dimensionality of the inner fully connected layer
	n = 6  # Number of layers in the decoder stack

	# Define the training parameters
	epochs = 400
	batch_size = 54
	beta_1 = 0.9
	beta_2 = 0.98
	epsilon = 1e-9
	dropout_rate = 0.1
	
	# Instantiate an Adam optimizer
	optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)
	
	# Prepare the training and test splits of the dataset
	dataset = PrepareDataset()
	trainProc, train_orig, dec_seq_max_length, dec_vocab_size = dataset(datasetFilename, level_style)
	
	print("Maximum sequence length: ", dec_seq_max_length)
	print("Vocabulary size: ", dec_vocab_size)
	
	# Prepare the dataset batches
	train_dataset = data.Dataset.from_tensor_slices(trainProc)
	train_dataset = train_dataset.batch(batch_size)
	
	# Include metrics monitoring
	train_loss = Mean(name = 'train_loss')
	train_accuracy = Mean(name = 'train_accuracy')
	
	if (trainAgain):
		# Create model
		training_model = TransformerModel(dec_vocab_size, dec_seq_max_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
		print("New Transformer Model instantiated")
	else:
		training_model = TransformerModel(dec_vocab_size, dec_seq_max_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
		
		# call the model before assigning the weights
		for step, train_batch in enumerate(train_dataset):
			# Define the encoder and decoder inputs, and the decoder output
			decoder_input = train_batch[:, :-1]
			training_model(decoder_input, training = True)
			break
		
		# load the weights for the model
		training_model.load_weights(model_save_path)
		print("Restored from {}".format(model_save_path))
		training_model.summary()


	###############################################################
	######################## Train ################################
 	###############################################################
 	
 	# start to take the time
	start_time = time()
 	
	for epoch in range(epochs):
		train_loss.reset_states()
		train_accuracy.reset_states()

		print("\nStart of epoch %d" % (epoch + 1))

		# Iterate over the dataset batches
		for step, train_batch in enumerate(train_dataset):
			# Define the encoder and decoder inputs, and the decoder output
			decoder_input = train_batch[:, :-1]
			decoder_output = train_batch[:, 1:]

			# do a trining step
			train_step(training_model, decoder_input, decoder_output)

			if (step % 8 == 0):
				print(f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
				
			if (train_accuracy.result() > 0.999):
				break


		# Print epoch number and loss value at the end of every epoch
		print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f" % (epoch + 1, train_loss.result(), train_accuracy.result()))

		# Save a checkpoint after every epochs
		training_model.save_weights(model_save_path)
		
		print("Saved checkpoint at epoch %d at path %s" % (epoch + 1, model_save_path))
 
print("Total time taken: %.2fs" % (time() - start_time))
	
	
	
