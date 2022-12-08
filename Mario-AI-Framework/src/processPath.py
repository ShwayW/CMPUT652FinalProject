# Code for processing the path that is outputted in the main gameplay loop of MarioGame.java into positionData.txt

import sys

def main(): 

	i = int(sys.argv[1])
	maxHeight = 16
	with open('positionData.txt') as f:
		first_line = f.readline().rstrip()
		first_length = int(first_line)
		levelData = [['-' for i in range(first_length)] for j in range(maxHeight)]

		for line in f.readlines():
			line = line.strip() 
			x,y = line.split(",")
			x = int(x)
			y = int(y)
			if(y<=maxHeight-1):
				levelData[y][x] = 'x'

	#folderName = "Data/Completionist/lvl-%d.txt"%(i)
	folderName = "../../output/completionist/hum-%d.txt"%(i)
	f = open(folderName, "w")
	for i in range(maxHeight): 
		level = levelData[i]
		if(i<maxHeight-1):
			levelString = "".join(level)+"\n"
		else:
			levelString = "".join(level)
		f.write(levelString)


if __name__ == '__main__':
    main()