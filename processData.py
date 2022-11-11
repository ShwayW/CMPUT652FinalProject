from lstm import getTokens, toOneHot

# Process the level and path data into a form that can be used by the LSTM 

def processPathAndLevel():

	level = "lvl-1s.txt"
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

	return oneHotLine, oneHotPath


if __name__ == '__main__':
	processPathAndLevel()