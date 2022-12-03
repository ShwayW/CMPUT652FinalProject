# Code for merging together the level and path data together

import sys

def main(): 

	i = int(sys.argv[1])
	#levelLocation = 'levels/original/'
	levelLocation = '../../output/speedrunner/'
	#levelName = 'lvl-%d.txt'%(i)
	levelName = 'output_0%dpath.txt'%(i)
	levelName2 = 'hum-%d.txt'%(i)
	dataSource = 'completionist/'

	with open(levelLocation+levelName) as fp:
		levelData = fp.readlines()
		levelData = [list(levelData[i].strip("\n")) for i in range(len(levelData))]
	
	#with open('Data/'+dataSource+levelName) as fp2:
	with open(levelLocation + levelName2) as fp2:
		pathData = fp2.readlines()
		pathData = [pathData[i].strip("\n") for i in range(len(pathData))]

	#print(levelData)
	#print('PATH DATA')
	#print(pathData)
	assert(len(levelData) == len(pathData))
	assert(len(levelData[0]) == len(pathData[0]))

	for i in range(len(levelData)):
		for j in range(len(levelData[0])):
			if(levelData[i][j] == 'x'):
				levelData[i][j] = '-'
			if(pathData[i][j] == 'x' and levelData[i][j] == '-'):
				levelData[i][j] = 'x'
	#f = open("Data/"+dataSource+levelName.strip(".txt")+"path"+".txt", "w")
	f = open(levelLocation+levelName2.strip(".txt")+"path"+".txt", "w")
	for i in range(len(levelData)): 
		level = levelData[i]
		if(i<len(levelData)-1):
			levelString = "".join(level)+"\n"
		else:
			levelString = "".join(level)
		f.write(levelString)

if __name__ == '__main__':
    main()