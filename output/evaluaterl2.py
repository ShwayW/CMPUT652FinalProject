import Levenshtein
import numpy as np

def main():
	editDistance = np.array([])
	folderName = 'rl2'
	humanFiles = {}
	humanStrings = {}
	levelFiles = {}
	levelStrings = {}
	strings = {}
	for i in range(1, 4):
		humanFiles[i] = '%s/exp6_human_lvl-1_%ds.txt'%(folderName, i)
		humanFile = open(humanFiles[i])
		humanStrings[i] = humanFile.readlines()[0].strip('\n')

		levelFiles[i] = '%s/exp6_lvl-2_lvl-1_%ds.txt'%(folderName, i)
		levelFile = open(levelFiles[i])
		levelStrings[i] = levelFile.readlines()[0].strip('\n')

		print(humanFiles[i])
		ld = Levenshtein.ratio(humanStrings[i], levelStrings[i])
		print(ld)
		editDistance = np.append(editDistance, [ld])

	
	

	print("Average Edit Distance", np.mean(editDistance))
	print("Standard Deviation Edit Distance", np.std(editDistance))
	# file4string = '%s/positionData_gail_lvl-1_%dpaths.txt'%(folderName)
	# file4 = open(file4string) 
	# string4 = file4.readlines()[0].strip('\n')


	# # print(len(string1))
	# # print(len(string2))
	# assert(len(string1) == len(string2))
	# assert(len(string2) == len(string3))
	# assert(len(string2) == len(string4))

	# # hd = hamming_distance(string1, string2)

	# #print("Hamming Distance", hd)

	# ld = Levenshtein.ratio(string1, string2)
	# print("Edit Distance Ratio Between Airl and Human", ld)

	# ld2 = Levenshtein.ratio(string2, string3)
	# print("Edit Distance Ratio between BC and Human", ld2)

	# ld3 = Levenshtein.ratio(string2, string4)
	# print("Edit Distance Ratio between Gail and Human", ld3)

	# speedrunnerNp = np.append(speedrunnerNp, [ld])
	# completionistNp = np.append(completionistNp, [ld2])
	# print("Average Speedrunner LD", np.mean(speedrunnerNp))
	# print("Average Completionist LD", np.mean(completionistNp))




if __name__ == '__main__':
	main()