from lstm import getTokens, toOneHot
import numpy as np

# Process the level and path data into a form that can be used by the LSTM 

def processPathAndLevel():

	level = "lvl-1sa.txt"
	rootDir = "Mario-AI-Framework/src/"
	levelDir = "levels/original/"
	pathDir = "Data/Justin/"


	levelData = open(rootDir+levelDir+level, "r")
	pathData = open(rootDir+pathDir+level, "r")

	levelLines = levelData.readlines()
	#levelLines = levelLines.strip()

	pathLines = pathData.readlines()
	#pathLines = pathLines.strip()

	tokens = getTokens(levelLines)
	oneHotLine = toOneHot(levelLines, tokens)

	pathTokens = getTokens(pathLines)
	oneHotPath = toOneHot(pathLines, pathTokens)

	fullDataset = [np.hstack([oneHotLine[0], oneHotPath[0]])]
	return fullDataset


if __name__ == '__main__':
	fullDataset = processPathAndLevel()
	print(fullDataset)