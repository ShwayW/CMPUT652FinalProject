import Levenshtein
import numpy as np

def main():
	editDistance = np.array([])
	folderName = 'rl2'
	filestrings = {}
	strings = {}
	filestrings[1] = '%s/final_lvl-1_1samp_humans.txt'%(folderName)
	file1 = open(filestrings[1])
	strings[1] = file1.readlines()[0].strip('\n')

	filestrings[2] = '%s/final_lvl-1_1samp_normals.txt'%(folderName)
	file2 = open(filestrings[2])
	strings[2] = file2.readlines()[0].strip('\n')

	filestrings[3] = '%s/final_lvl-1_1samp_random_pass2s.txt'%(folderName)
	file3 = open(filestrings[3]) 
	strings[3] = file3.readlines()[0].strip('\n')


	filestrings[4] = '%s/final_lvl-1_1samp_randomsticky_passs.txt'%(folderName)
	file4 = open(filestrings[4]) 
	strings[4] = file4.readlines()[0].strip('\n')

	filestrings[5] = '%s/final_lvl-1_1samp_sticky_pass2s.txt'%(folderName)
	file5 = open(filestrings[5]) 
	strings[5] = file5.readlines()[0].strip('\n')


	for i in range(2,6):
		ld = Levenshtein.ratio(strings[1], strings[i])
		print(filestrings[i])
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