# Code that works on taking the user's path data and outputs it in 


def main():
	stringIndex = 0

	maxHeight = 16

	top = True

	source = "output/speedrunner/output_08.txt"
	destination = "output/speedrunner/output_08path.txt"
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
