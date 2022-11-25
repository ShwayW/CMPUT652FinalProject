# Code that works on taking the user's path data and outputs it in 


def main():

	reversible = False 
	#directory = "../../output/"
	directory = "Data/Speedrunner/"
	#directory = "Data/Justin/"
	maxHeight = 16

	baseLevel = "lvl-10paths"
	#directory = "levels/original/"
	if(reversible):
		level = baseLevel+"a"
		top = False	
		stringIndex = maxHeight-1
	else: 
		level = baseLevel
		top = True	
		stringIndex = 0

	source = directory+level+".txt"
	destination = directory+level+"path.txt"
	f = open(source, "r")
	fs = open(destination, "w")

	line = f.readline()
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
	fs.write(finalString)
	f.close()
	fs.close()



if __name__ == '__main__':
    main()
