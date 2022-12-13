# Shway Wang
# code modified from https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking

from pickle import load
from tensorflow import Module
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64, TensorArray, argmax, newaxis, transpose, data
from model import TransformerModel
from prepare_dataset import PrepareDataset
from PIL import Image
import time
import os
import glob

def visualize(outputFileName):
	#Load the set of all sprites
	sprites = {}
	for filename in glob.glob(os.path.join("sprites", "*.png")):
		im = Image.open(filename)
		splits = filename.split("/")
		name = splits[-1][:-4]
		sprites[name] = im.convert('RGBA')

	#This gives the mapping between the tile values and the associated sprite
	visualization = {}
	visualization["S"] = "brick"
	visualization["?"] = "exclamationBox"
	visualization["Q"] = "exclamationBoxEmpty"
	visualization["E"] = "enemy"
	visualization["g"] = "enemy"
	visualization["U"] = "exclamationBox"
	visualization["?"] = "exclamationBox"
	visualization["#"] = "pyramind"
	visualization["<"] = "bushTopLeft"
	visualization[">"] = "bushTopRight"
	visualization["["] = "bushLeft"
	visualization["]"] = "bushRight"
	visualization["o"] = "coin"
	visualization["B"] = "arrowTop"
	visualization["b"] = "arrowBottom"
	visualization["x"] = "mario"
	visualization["L"] = "exclamationBox"
	visualization["@"] = "exclamationBox"
	visualization["T"] = "pipe" 
	visualization["C"] = "brick"
	visualization["M"] = "mario"
	visualization["k"] = "greenkoopa"
	visualization["r"] = "redkoopa"
	visualization["F"] = "finishline"
	visualization["!"] = "exclamationBox"
	visualization["t"] = "pipe"
	visualization["y"] = "spiky"
	visualization["R"] = "redkoopa"
	visualization["K"] = "paratroopa"
	visualization["*"] = "bulletbill"
	visualization["|"] = "backgroundtile"
	visualization["%"] = "backgroundtile"
	
	# This reads in the level
	level = {}
	with open(outputFileName + ".txt") as fp:
		y = 0
		for line in fp:
			level[y] = line.strip("\n")
			y += 1
			
	# Multiply by 18 here as each of the sprites is 18*18
	# This creates an initially blank image for the level
	image = Image.new("RGB", (18 * len(level[0]), 18 * len(level.keys())), color=(223, 245, 244))
	
	# This loads the level image's pixels so we can edit them
	pixels = image.load()
	maxY = len(level.keys())
	maxX = len(level[0])


	for y in range(0, maxY):
		for x in range(0, maxX):
			imageToUse = None
			if level[y][x] in visualization.keys():
				imageToUse = sprites[visualization[level[y][x]]]
			elif level[y][x]=="X":
				#Rules we've added to ensure the correct sprite is used
				if y==maxY-2:
					imageToUse = sprites["groundTop"]
				elif y==maxY-1:
					#Check if we have a solid tile above this and change which sprite we use if so
					if level[y-1][x]=="X":
						imageToUse = sprites["groundBottom"]
					else:
						imageToUse = sprites["groundTop"]
				else:
					imageToUse = sprites["stair"]

			elif level[y][x]!='-':
				print(level[y][x])
			if not imageToUse == None:
				#If we have a sprite (imageToUse) copy its pixels over
				pixelsToUse = imageToUse.load()
				for x2 in range(0, 18):
					for y2 in range(0, 18):
						if pixelsToUse[x2, y2][3]>0:
							pixels[x * 18 + x2, y * 18 + y2] = pixelsToUse[x2, y2][0:-1]
	#Save the output to a jpeg
	image.save(outputFileName + ".png", "PNG")
	
	

def snakeToPath(sourceFileName, destinationFileName):
	stringIndex = 0
	maxHeight = 16
	top = True

	sourceFile = open(sourceFileName + '.txt', "r")
	destFile = open(destinationFileName + '.txt', "w")

	line = sourceFile.readline()
	line = line.strip("\n")

	outputStrings = ["" for j in range(maxHeight)]

	for i in range(len(line)):
		if(top):
			outputStrings[stringIndex] += line[i]
			if(stringIndex< maxHeight-1):
				stringIndex = stringIndex+1 
			else: 
				top = False 
		else:
			outputStrings[stringIndex] += line[i]
			if(stringIndex):
				stringIndex = stringIndex - 1 
			else:
				top = True 

	finalString = "\n".join(outputStrings)
	destFile.write(finalString)
	sourceFile.close()
	destFile.close()
	

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

	def __call__(self, prompt, maxInputLen, level_style):
		# Load decoder tokenizer
		if (level_style == 'speedrunner'):
			dec_tokenizer = self.load_tokenizer('speedrunner_levels_tokenizer.pkl')
		elif (level_style == 'completionist'):
			dec_tokenizer = self.load_tokenizer('completionist_levels_tokenizer.pkl')
		
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
	level_style = 'speedrunner'
	#level_style = 'completionist'
	
	if (level_style == 'speedrunner'):
		datasetFilename = './speedrunner_levels.pkl'
		model_save_path = './models/transformer_speedrunner.h5'
		destination_path = 'output_transformer/speedrunner/'
	elif (level_style == 'completionist'):
		datasetFilename = './completionist_levels.pkl'
		model_save_path = './models/transformer_completionist.h5'
		destination_path = 'output_transformer/completionist/'

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
	maxInputLen = 320
	
	# Random 32-tile strings
	'''
	K#SF!L1-E!Eo-g|?EMFMkEbUrb!ooMK@
	M@2TS|EUEU#XxU@bo@1*#-|T1QrrX-E|
	1?2@C%!rFk|ES!LkXBtBbB*x-LTboFto
	2UFXt@xgyr@k#MM2XRytTFCC2Er2CCFy
	QFkt@SMbk-byT#|Qx@*r1F%x?y!|-xtX
	F#-E|Q?QtQ%2g2bBoCFxBKL%ML*kUCQy
	rTL1x11RU|@gEb-2U-ByrBro*yo!1oCb
	KR1LCC?1Rro@?KR%E|yXy##!X*b-rCx%
	g@XETTXbRy|Eo2@RESX%%FR%MF@bBM@b
	22B-o@o-KF#t-RkMytLBLF!!%?L??LC!
	'''
	
	# the prompt, each is 32-tile-long
	prompts =  ['--------------XXXX--------------', # lvl-1
				'---SSSSSSSSSSSXXXX--------------', # lvl-2
				'-------------xXXXXx-------------', # lvl-3
				'----------xxxMXXXXxxx-----------', # lvl-4
				'--------------XXXXM-------------', # lvl-5
				'-------------xXXXXxxx-----------', # lvl-6
				'--xSSSSSSSSSSSXXXX----------xx--', # lvl-8
				'-----------xxxXXXX---x----------', # lvl-9
				'------------xxXXXX--xx----------', # lvl-11
				'----------xxxxXXXXx-------------', # lvl-12
				# random prompts
				'K#SF!L1-E!Eo-g|?EMFMkEbUrb!ooMK@', # random
				'2UFXt@xgyr@k#MM2XRytTFCC2Er2CCFy',
				'1?2@C%!rFk|ES!LkXBtBbB*x-LTboFto',
				'M@2TS|EUEU#XxU@bo@1*#-|T1QrrX-E|',
				'QFkt@SMbk-byT#|Qx@*r1F%x?y!|-xtX',
				'F#-E|Q?QtQ%2g2bBoCFxBKL%ML*kUCQy',
				'rTL1x11RU|@gEb-2U-ByrBro*yo!1oCb',
				'KR1LCC?1Rro@?KR%E|yXy##!X*b-rCx%',
				'g@XETTXbRy|Eo2@RESX%%FR%MF@bBM@b',
				'22B-o@o-KF#t-RkMytLBLF!!%?L??LC!']
	
	# Desired generation length
	genLen = 3200
	
	# segment length
	segLen = 320
	
	# Use traned model
	useTrainedModel = True

	# Prepare the training and test splits of the dataset
	dataset = PrepareDataset()
	trainProc, train_orig, dec_seq_max_length, dec_vocab_size = dataset(datasetFilename, level_style)
	
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
	
	#############################################
	# The batch inference loop
	#############################################
	for promptI in range(len(prompts)):
		start = time.time()
	
		# get the current prompt
		prompt = prompts[promptI]
		
		# preprocess the prompt to char by char
		preced_prompt = []
		for i in range(len(prompt)):
			preced_prompt.append(prompt[i])
		
		# predict the level
		entirePred = ''
		for segI in range(0, genLen, segLen):
			inference = Inference(inferencing_model, segLen)
			pred = inference([preced_prompt], maxInputLen, level_style)
			entirePred += pred
			
			# preprocess the prompt to char by char
			preced_prompt = []
			for i in range(len(pred[-32:])):
				preced_prompt.append(pred[-32:][i])
			
		
		# write the level to the source file in Snake-path format
		sourceFileName = destination_path + 'output_' + str(promptI + 1)
		fs = open(sourceFileName + '.txt', "w")
		fs.write(entirePred)
		fs.close()

		# convert the Snake format to path format
		destFileName = destination_path + 'output_' + str(promptI + 1) + 'path'
		snakeToPath(sourceFileName, destFileName)
		visualize(destFileName)
		
		end = time.time()
		print("time it took to generate one level: ", end - start)
		
		
		
	
	
	
	
	
	
	
	
