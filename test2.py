import numpy 
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D

def readDataSet(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = numpy.zeros((numberOfLines-1,3))

    classLabelVector = []
    classColorVector = []

    index = 0

    for line in fr.readlines(): 
        if index != 0:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index-1,:] = listFromLine[1:4]
            classLabel = listFromLine[4]

            if classLabel == "Buero":
                color = 'yellow'

            elif classLabel == "Wohnung":
                color = 'red'
            else:
                color = 'blue'

            classLabelVector.append(classLabel)
            classColorVector.append(color)

        index += 1

    return returnMat, classLabelVector, classColorVector

dataSet, classLabelVector, classColorVector = readDataSet("Daten.txt")
print(dataSet)
print(classLabelVector)
print(classColorVector)

