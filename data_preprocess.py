import string
import re
from pickle import load, dump
from unicodedata import normalize
from numpy import array
from numpy.random import rand
from numpy.random import shuffle

def textDataPreProc(path):
	# get all the lines
	lines = []
	path += '{}'
	for i in range(1, 402):
	    if (i < 10):
	        index = "00{}.txt".format(i)
	    elif (i < 100):
	        index = "0{}.txt".format(i)
	    else:
	        index = "{}.txt".format(i)
	    with open(path.format(index), 'r') as f:
	        lines.extend(f.readlines())

	# Remove new lines and empty lines
	lines = [line for line in lines if line != '\n' or '']

	# Strip all lines
	lines = [line.strip() for line in lines]

	# remove lines that are less than 10 tokens
	doc = [line for line in lines if (len(line) > 10 and len(line) < 500)]
	
	# return the one hot vectors of training and test sets and the set of tokens
	return doc
	
	
# clean a list of lines
def clean_doc(lines):
	cleaned = []
	
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		# normalize unicode characters
		line = normalize('NFD', line).encode('ascii', 'ignore')
		line = line.decode('UTF-8')
		
		# remove non-printable chars form each token
		line = [re_print.sub('', w) for w in line]
		
		# store the cleaned line
		cleaned.append(line)
	return array(cleaned)


# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)


if (__name__ == "__main__"):
	# save path
	savePath = 'news.pkl'

	# load dataset
	fileFolder = './bbc-news-summary/BBC News Summary/News Articles/tech/'
	doc = textDataPreProc(fileFolder)

	# clean sentences
	cleaned_doc = clean_doc(doc)

	# spot check
	for i in range(20):
		print(cleaned_doc[i])

	print(len(cleaned_doc))

	# random shuffle
	shuffle(cleaned_doc)

	# save
	save_clean_data(cleaned_doc, savePath)
	
	
