import Levenshtein
import numpy as np

def main():
	speedrunnerNp = np.array([])
	completionistNp = np.array([])
	for i in range(1, 8):
		if(i!=10): 
			folderName = 'speedrunner'
			print(i)
			print("Level %d"%(i))
			file1string = '%s/lvl-%dpaths.txt'%(folderName, i)
			file1 = open(file1string)
			string1 = file1.readlines()[0].strip('\n')

			file2string = '%s/output_0%d.txt'%(folderName, i)
			file2 = open(file2string)
			string2 = file2.readlines()[0].strip('\n')

			file3string = '%s/hum-%dpaths.txt'%(folderName, i)
			file3 = open(file3string) 
			string3 = file3.readlines()[0].strip('\n')

			# print(len(string1))
			# print(len(string2))
			assert(len(string1) == len(string2))
			assert(len(string2) == len(string3))

			# hd = hamming_distance(string1, string2)

			#print("Hamming Distance", hd)
			
			ld = Levenshtein.ratio(string1, string2)
			print("Edit Distance Ratio Between Predicted Path and Speedrunner", ld)

			ld2 = Levenshtein.ratio(string2, string3)
			print("Edit Distance Ratio between Predicted Path and Completionist", ld2)

			speedrunnerNp = np.append(speedrunnerNp, [ld])
			completionistNp = np.append(completionistNp, [ld2])
	print("Average Speedrunner LD", np.mean(speedrunnerNp))
	print("Average Completionist LD", np.mean(completionistNp))




if __name__ == '__main__':
	main()