import Levenshtein


# Code Source: https://en.wikipedia.org/wiki/Hamming_distance
def hamming_distance(string1, string2):
	dist_counter = 0
	for n in range(len(string1)):
		if string1[n] != string2[n]:
			dist_counter += 1
	return dist_counter


def main():
	for i in range(1, 3):
		print("Level %d"%(i))
		file1string = 'speedrunner/lvl-%dpaths.txt'%(i)
		file1 = open(file1string)
		string1 = file1.readlines()[0].strip('\n')

		file2string = 'speedrunner/output_0%d.txt'%(i)
		file2 = open(file2string)
		string2 = file2.readlines()[0].strip('\n')

		assert(len(string1) == len(string2))

		hd = hamming_distance(string1, string2)

		print("Hamming Distance", hd)
		
		ld = Levenshtein.distance(string1, string2)
		print("Edit Distance", ld)




if __name__ == '__main__':
	main()