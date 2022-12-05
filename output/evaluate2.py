import Levenshtein
import numpy as np

def main():
	bcNp = np.array([])
	airlNp = np.array([])    
	gailNp = np.array([])
	for i in range(1, 4):
		folderName = 'rl'
		print(i)
		print("Level %d"%(i))
		file1string = '%s/positionData_airl_lvl-1_%dpaths.txt'%(folderName, i)
		file1 = open(file1string)
		string1 = file1.readlines()[0].strip('\n')

		file2string = '%s/positionData_human_lvl-1_%dpaths.txt'%(folderName, i)
		file2 = open(file2string)
		string2 = file2.readlines()[0].strip('\n')

		file3string = '%s/positionData_bc_lvl-1_%dpaths.txt'%(folderName, i)
		file3 = open(file3string) 
		string3 = file3.readlines()[0].strip('\n')
		
		file4string = '%s/positionData_gail_lvl-1_%dpaths.txt'%(folderName, i)
		file4 = open(file4string) 
		string4 = file4.readlines()[0].strip('\n')
		# print(len(string1))
		# print(len(string2))
		assert(len(string1) == len(string2))
		assert(len(string2) == len(string3))
		assert(len(string2) == len(string4))

		# hd = hamming_distance(string1, string2)

		#print("Hamming Distance", hd)

		ld = Levenshtein.ratio(string1, string2)
		print("Edit Distance Ratio Between Airl and Human", ld)

		ld2 = Levenshtein.ratio(string2, string3)
		print("Edit Distance Ratio between BC and Human", ld2)

		ld3 = Levenshtein.ratio(string2, string4)
		print("Edit Distance Ratio between Gail and Human", ld3)

		# speedrunnerNp = np.append(speedrunnerNp, [ld])
		# completionistNp = np.append(completionistNp, [ld2])
		# print("Average Speedrunner LD", np.mean(speedrunnerNp))
		# print("Average Completionist LD", np.mean(completionistNp))




if __name__ == '__main__':
	main()