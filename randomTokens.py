from data_preprocess import levelDataPreProc
import random
import pickle

def main():
    lines = levelDataPreProc('./Mario-AI-Framework/src/Data/Speedrunner/')
    lines2 = levelDataPreProc('./Mario-AI-Framework/src/Data/Completionist/')

    allTokens = []

    for i in range(15):
        line = lines[i]
        for c in line:
            if c not in allTokens:
                allTokens.append(c)
    randomStrings = []
    for i in range(10): 
        randomString = "".join(random.choices(allTokens, k = 32))

    file = open('randomStrings', 'wb')
    pickle.dump(randomStrings, file)


if __name__ == '__main__':
    main()