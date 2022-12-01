# Code that works on taking the user's path data and outputs it in 

import sys 

def main():
	#directory = "Data/Completionist/"
	#directory = "Data/Speedrunner/"
	#directory = "levels/original/"
	directory = "../../output/completionist/"
	i = int(sys.argv[1])
	level = "lvl-%dpath"%(i)

	# Load in the input path data as a 2d array and output it in a snake format in two different ways
	source = directory+level+".txt"

	reversible = False 

	if(reversible):
		destination = directory+level+"sa.txt"
		top = False	
	else:
		destination = directory+level+"s.txt"
		top = True

	f = open(source, "r")
	fs = open(destination, "w")

	lines = f.readlines()


	maxHeight = 16
	maxLength = len(lines[maxHeight-1])

	outputString = ""
	for i in range(maxLength):
		if(top):
			# Go from the top of the array down to the bottom
			for j in range(maxHeight):
				outputString += lines[j][i]
			# Switch it to create a snake like pattern 
			top = False
		else:
			for j in range(maxHeight-1, -1, -1):
				outputString += lines[j][i]
			# Switch it to create a snake like pattern
			top = True 
	fs.write(outputString)
	f.close()
	fs.close()



if __name__ == '__main__':
    main()
