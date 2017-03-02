import numpy as np
import subprocess

'''
#read file for algo.
def readfile(matchFile):
    fileID = open(matchFile,'r')
    readlines = fileID.readlines()

    line = []
    first = readlines[0].split(' ')
    #imagePoint1 = np.array(map(float, first[:2]))
    #imagePoint2 = np.array(map(float, first[2:4]))
    imagePoint1 = np.array(first[:2], dtype=np.float32)
    imagePoint2 = np.array(first[2:4], dtype=np.float32)

    for lines in readlines[1:]:
        words = lines.split(' ')
        if words[0] == '\n':
            break
        #Point1 = np.atleast_3d(map(float, words[:2]))
        #Point2 = np.atleast_3d(map(float, words[2:4]))
        #Point1 = np.array(map(float, words[:2]))
        #Point2 = np.array(map(float, words[2:4]))
        Point1 = np.array(words[:2], dtype=np.float32)
        Point2 = np.array(words[2:4], dtype=np.float32)
        imagePoint1 = np.vstack((imagePoint1,Point1))
        imagePoint2 = np.vstack((imagePoint2,Point2))
    return (imagePoint1, imagePoint2)


imp1, imp2 = readfile('out.txt')
print(imp2[1])

print('ya')
'''

bash1 = 'cd'
subprocess.call(bash1, shell=True)