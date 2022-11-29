# Code for merging together the level and path data together

import sys

def main(): 

	i = int(sys.argv[1])
	levelLocation = 'levels/original/'
	levelName = 'lvl-%d.txt'%(i)
	dataSource = 'Completionist/'

	with open(levelLocation+levelName) as fp:
		levelData = fp.readlines()
		levelData = [list(levelData[i].strip("\n")) for i in range(len(levelData))]
	
	with open('Data/'+dataSource+levelName) as fp2:
		pathData = fp2.readlines()
		pathData = [pathData[i].strip("\n") for i in range(len(pathData))]


	assert(len(levelData) == len(pathData))
	assert(len(levelData[0]) == len(pathData[0]))

	for i in range(len(levelData)):
		for j in range(len(levelData[0])):
			if(pathData[i][j] == 'x' and levelData[i][j] == '-'):
				levelData[i][j] = 'x'
	f = open("Data/"+dataSource+levelName.strip(".txt")+"path"+".txt", "w")

	for i in range(len(levelData)): 
		level = levelData[i]
		if(i<len(levelData)-1):
			levelString = "".join(level)+"\n"
		else:
			levelString = "".join(level)
		f.write(levelString)

if __name__ == '__main__':
    main()