from pickle import load, dump
from numpy.random import shuffle
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64
 
 
class PrepareDataset:
	def __init__(self, **kwargs):
		super(PrepareDataset, self).__init__(**kwargs)
		self.n_sentences = 41936  # Number of sentences to include in the dataset
		self.train_split = 1.0  # Ratio of the training data split
 
	# Fit a tokenizer
	def create_tokenizer(self, dataset):
		tokenizer = Tokenizer(lower = False)
		tokenizer.fit_on_texts(dataset)
		return tokenizer
 
	def find_seq_max_length(self, dataset):
		return max(len(seq) for seq in dataset)
 
	def find_vocab_size(self, tokenizer, dataset):
		tokenizer.fit_on_texts(dataset)
		print(tokenizer.word_index)
		return len(tokenizer.word_index) + 1
 
	def __call__(self, filename, **kwargs):
		# Load a clean dataset
		clean_dataset = load(open(filename, 'rb'))

		# Reduce dataset size
		dataset = clean_dataset[:self.n_sentences]
		
		# Random shuffle the dataset
		shuffle(dataset)

		# Split the dataset
		train = list(dataset[:int(self.n_sentences * self.train_split)])
		
		# convert to list
		for i in range(len(train)):
			train[i] = list(train[i])

		# Prepare tokenizer for the decoder input
		dec_tokenizer = self.create_tokenizer(train)
		
		# store the tokenizer
		dump(dec_tokenizer, open('levels_tokenizer.pkl', 'wb'))
		
		# compute the decoder max sequence length and vocabulary size
		dec_seq_max_length = self.find_seq_max_length(train)
		dec_vocab_size = self.find_vocab_size(dec_tokenizer, train)

		# Encode and pad the input sequences
		trainProc = dec_tokenizer.texts_to_sequences(train)
		trainProc = pad_sequences(trainProc, maxlen = dec_seq_max_length, padding = 'post')
		trainProc = convert_to_tensor(trainProc, dtype = int64)

		return trainProc, train, dec_seq_max_length, dec_vocab_size
		

