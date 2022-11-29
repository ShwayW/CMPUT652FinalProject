import string
import re
from pickle import load, dump
from unicodedata import normalize
from numpy import array
from numpy.random import rand
from numpy.random import shuffle

def levelDataPreProc(path):
	# get all the lines
	lines = []
	path += 'lvl-{}paths.txt'
	for i in range(1, 16):
	    index = "{}".format(i)
	    with open(path.format(index), 'r') as f:
	        lines.extend(f.readlines())

	# Remove new lines and empty lines
	lines = [line for line in lines if line != '\n' or '']

	# Strip all lines
	lines = [line.strip() for line in lines]
	
	# return the one hot vectors of training and test sets and the set of tokens
	return lines
	
	
# clean a list of lines
def split_to_chunks(lines, chunkLen):
	chunks = []
	
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	
	# for each line in lines
	for line in lines:
		# normalize unicode characters
		line = normalize('NFD', line).encode('ascii', 'ignore')
		line = line.decode('UTF-8')
		
		for chunkI in range(0, len(line) - chunkLen):
			# store the cleaned line
			chunk = [re_print.sub('', w) for w in line[chunkI:chunkLen + chunkI]]
			chunks.append(chunk)
	return array(chunks)


# save a list of chunks to file
def save_clean_data(chunks, filename):
	dump(chunks, open(filename, 'wb'))
	print('Saved: %s' % filename)


if (__name__ == "__main__"):
	# save path
	savePath = 'levels.pkl'

	# load dataset
	fileFolder = './Mario-AI-Framework/src/Data/Speedrunner/'
	
	# the length of each chunk of level to be stored
	chunkLen = 320
	
	levels = levelDataPreProc(fileFolder)

	# clean sentences
	chunked_levels = split_to_chunks(levels, chunkLen)

	# spot check
	for i in range(20):
		print(chunked_levels[i])

	print(len(chunked_levels))

	# random shuffle
	shuffle(chunked_levels)

	# save
	save_clean_data(chunked_levels, savePath)
	
	
