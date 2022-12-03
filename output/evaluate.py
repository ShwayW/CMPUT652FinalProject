import Levenshtein


# Code Source: https://en.wikipedia.org/wiki/Hamming_distance
def hamming_distance(string1, string2):
	dist_counter = 0
	for n in range(len(string1)):
		if string1[n] != string2[n]:
			dist_counter += 1
	return dist_counter


def main():
	for i in range(1, 4):
		print("Level %d"%(i))
		file1string = 'speedrunner/lvl-%dpaths.txt'%(i)
		file1 = open(file1string)
		string1 = file1.readlines()[0].strip('\n')

		file2string = 'speedrunner/output_0%d.txt'%(i)
		file2 = open(file2string)
		string2 = file2.readlines()[0].strip('\n')

		file3string = 'speedrunner/hum-%dpaths.txt'%(i)
		file3 = open(file3string) 
		string3 = file3.readlines()[0].strip('\n')

		assert(len(string1) == len(string2))
		assert(len(string2) == len(string3))

		hd = hamming_distance(string1, string2)

		#print("Hamming Distance", hd)
		
		ld = Levenshtein.ratio(string1, string2)
		print("Edit Distance Ratio Between Predicted Path and Speedrunner", ld)

		ld2 = Levenshtein.ratio(string2, string3)
		print("Edit Distance Ratio between Predicted Path and Completionist", ld2)




if __name__ == '__main__':
	main()