import re
import numpy as np

def main():
    dataset1 = np.array([])
    dataset2 = np.array([])
    dataset3 = np.array([])
    dataset4 = np.array([])

    dataset1a = np.array([])
    dataset2a = np.array([])
    dataset3a = np.array([])
    dataset4a = np.array([])

    dataset1b = np.array([])
    dataset2b = np.array([])
    dataset3b = np.array([])
    dataset4b = np.array([])

    for i in range(1,8):
        if i!=5:
            fileName1 = 'speedrunner/hum_0%dmetadata.txt'%(i)
            fileName2 = 'speedrunner/output_0%dmetadata.txt'%(i)
            fileName3 = 'completionist/hum_0%dmetadata.txt'%(i)
            fileName4 = 'completionist/output_0%dmetadata.txt'%(i)
            file1 = open(fileName1, 'r')
            file2 = open(fileName2, 'r')
            file3 = open(fileName3, 'r')
            file4 = open(fileName4, 'r')

            file1string = file1.readlines() 
            file2string = file2.readlines()
            file3string = file3.readlines()
            file4string = file4.readlines()

            lines1 = file1string[1]
            lines2 = file2string[1]
            lines3 = file3string[1]
            lines4 = file4string[1]

            numbers1 = re.findall(r'\d+', lines1)
            numbers2 = re.findall(r'\d+', lines2)
            numbers3 = re.findall(r'\d+', lines3)
            numbers4 = re.findall(r'\d+', lines4)


            dataset1 = np.append(dataset1, int(numbers1[1]))
            dataset2 = np.append(dataset2, int(numbers2[1]))
            dataset3 = np.append(dataset3, int(numbers3[1]))
            dataset4 = np.append(dataset4, int(numbers4[1]))


            dataset1a = np.append(dataset1a, 200-int(numbers1[2]))
            dataset2a = np.append(dataset2a, 200-int(numbers2[2]))
            dataset3a = np.append(dataset3a, 200-int(numbers3[2]))
            dataset4a = np.append(dataset4a, 200-int(numbers4[2]))

            lines1b = file1string[3]
            lines2b = file2string[3]
            lines3b = file3string[3]
            lines4b = file4string[3]

            numbers1b = re.findall(r'\d+', lines1b)
            numbers2b = re.findall(r'\d+', lines2b)
            numbers3b = re.findall(r'\d+', lines3b)
            numbers4b = re.findall(r'\d+', lines4b)

            dataset1b = np.append(dataset1b, int(numbers1b[0]))
            dataset2b = np.append(dataset2b, int(numbers2b[0]))
            dataset3b = np.append(dataset3b, int(numbers3b[0]))
            dataset4b = np.append(dataset4b, int(numbers4b[0]))

    print("Average Number of Coins Collected Completionist Player Speedrunner Generator", np.mean(dataset1))
    print("Average Number of Coins Collected Speedrunner Player Speedrunner Generator", np.mean(dataset2))
    print("Average Number of Coins Collected Completionist Player Completionist Generator", np.mean(dataset3))
    print("Average Number of Coins Collected Speedrunner Player Completionist Generator", np.mean(dataset4))
    print("\n")
    print("Average time Completionist Player Speedrunner Generator", np.mean(dataset1a))
    print("Average time Speedrunner Player Speedrunner Generator", np.mean(dataset2a))
    print("Average time Completionist Player Completionist Generator", np.mean(dataset3a))
    print("Average time Speedrunner Player Completionist Generator", np.mean(dataset4a))
    print("\n")
    print("Average enemies killed Completionist Player Speedrunner Generator", np.mean(dataset1b))
    print("Average enemies killed Speedrunner Player Speedrunner Generator", np.mean(dataset2b))
    print("Average enemies killed Completionist Player Completionist Generator", np.mean(dataset3b))
    print("Average enemies killed Speedrunner Player Completionist Generator", np.mean(dataset4b))
            
if __name__ == '__main__':
    main()