import os
import sys
import csv
import math
sys.path.append(os.path.abspath("./libsvm/python"))
# print(sys.path)
from svm import *
from svmutil import *
import numpy as np
import matplotlib.pyplot as plt
import random

# dataset description:
# 
#1. vendor name: 30 
#(adviser, amdahl,apollo, basf, bti, burroughs, c.r.d, cambex, cdc, dec, 
#dg, formation, four-phase, gould, honeywell, hp, ibm, ipl, magnuson, 
#microdata, nas, ncr, nixdorf, perkin-elmer, prime, siemens, sperry, 
#sratus, wang) 
#2. Model Name: many unique symbols 
#3. MYCT: machine cycle time in nanoseconds (integer) 
#4. MMIN: minimum main memory in kilobytes (integer) 
#5. MMAX: maximum main memory in kilobytes (integer) 
#6. CACH: cache memory in kilobytes (integer) 
#7. CHMIN: minimum channels in units (integer) 
#8. CHMAX: maximum channels in units (integer) 
#9. PRP: published relative performance (integer) 
#10. ERP: estimated relative performance from the original article (integer)

# file = open("data.dat", "r") 
file = csv.reader(open('data.dat', 'rb'), delimiter=',')


# pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose

def issetOption(opt):
    return sys.argv.count(opt)

def getOption(opt, default = None):
    if issetOption(opt):
        return sys.argv[sys.argv.index(opt)+1]
    return default
    
def generateColumnsMatrix(columnsCount):
    columns = [];
    for i in range(columnsCount):
        columns.append([])
    return columns

def castToFloat(x):
    return float(x);

def findNormalizationDivider(maxV):
    powerOfTen = 0;
    while (maxV - (maxV % (10 ** powerOfTen))) / (10 ** powerOfTen) > 0:
        powerOfTen = powerOfTen + 1
    # print powerOfTen
    # print maxV
    return 10 ** powerOfTen

# return tuple (newVector, inputValuesRange, outputValuesRange, inputOffset)
def normalize(X, low=0, high=1, minX=None, maxX=None):
    X = np.asanyarray(X)
    if minX is None:
        minX = np.min(X)
    if maxX is None:
        maxX = np.max(X)
    # Normalize to [0...1]. 
    X = X - minX
    X = X / (maxX - minX)
    # Scale to [low...high].
    X = X * (high-low)
    X = X + low
    return X, maxX - minX, high - low, minX

def loadDataFromOpenedFile():
    global headers, attributes, columns, columnsNormalizationParams, lines, classes
    COLUMNS_COUNT = 8
    headers = ["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"]
    attributes = []
    columns = generateColumnsMatrix(COLUMNS_COUNT)
    columnsNormalizationParams = [];
    lines = []
    # read and convert data
    # randomize 
    for line in file:
        lines.append(line)
    # if(randomize):
    #     random.shuffle(lines)

    for idx, line in enumerate(lines): 
        # line = line.replace("\n", "").split(",")
        
        for key, item in enumerate(line[2:COLUMNS_COUNT]):
            # print key
            columns[key].append(float(item));
        # columns 0 and 1 are useless for machine learning, so we ignore them
        # column 9 is indeed the same as column 8 so we also ignore it
        currentAttribute = list(map(castToFloat, line[2:COLUMNS_COUNT]))
        attributes.append(currentAttribute)
        # print currentAttribute
    
    # normalize data
    # we normalize only 6 columns
    for idx, column in enumerate(columns[:5]):
        normalized = normalize(column)
        columnsNormalizationParams.append((normalized[1], normalized[2], normalized[3]))
        normalized = normalized[0]
        columns[idx] = list(normalized)
        # divider = findNormalizationDivider(max(column))
        for key, line in enumerate(attributes):
            attributes[key][idx] = normalized[key];
        #     attributes[key][idx] = attributes[key][idx] / divider;

    classes = columns[5:6][0]

def runWithParams(nu = None, gamma = None, compact = 0, withTest = 0):
    print classes
    print max(classes)
    print min(classes)
    print '\n\n\n\n\n\n\n\n\n\n'
    print 'count of records: ', len(attributes)

    ##
    #  Main Part
    ##
    def rateAndDraw(inputSet, outputSet, label = 'diff1'):
        diff = []
        for idx, el in enumerate(inputSet):
            err = math.fabs(inputSet[idx] - outputSet[idx][0])
            diff.append(err)

        plt.plot(range(len(inputSet)), diff)

    def learnAndTest(firstTestIndex, lastTestIndex):
        la = len(attributes)

        learn_classes = classes[0:firstTestIndex] + classes[lastTestIndex:la]
        learn_attributes = attributes[0:firstTestIndex] + attributes[lastTestIndex:la]

        test_classes = classes[firstTestIndex:lastTestIndex]
        test_attributes = attributes[firstTestIndex:lastTestIndex]

        param=svm_parameter("-q")
        param.svm_type = NU_SVR
        param.kernel_type = RBF

        if(gamma != None):
            param.gamma = gamma
        
        if(nu != None):
            param.nu = nu

        # param.cross_validation=1
        param.nr_fold=10

        problem = svm_problem(learn_classes, learn_attributes)
        # param = svm_parameter(kernel_type = RBF, C = 10)

        model = svm_train(problem, param)

        # testing

        print '\n\nGamma: ', param.gamma
        print '\nNu: ', param.nu

        p_lbl, p_acc, p_prob = svm_predict(learn_classes, learn_attributes, model)
        
        # verbose
        # if issetOption('-v'):
            # print p_lbl     

        print p_acc
        # verbose
        if issetOption('-v'):
            rateAndDraw(learn_classes, p_prob)
            if not compact:
                print learn_classes
                print p_prob

        if withTest:

            p_lbl, p_acc, p_prob = svm_predict(test_classes, test_attributes, model)
            
            # verbose
            # if issetOption('-v'):
                # print p_lbl     

            print p_acc
            # verbose
            if issetOption('-v') and not compact:
                print test_classes
                print p_prob
            
        

    inter = int(0.2*len(attributes))


    plt.figure()
    
    for el in range(4):
        learnAndTest(el * inter, (el+1)*inter);

    learnAndTest(4*inter, len(attributes));
    
    plt.show()

    # we are creating our intelligence model


    # Ostatnich dwoch wartosci nie normalizujemy, a ostatnia mozna odrzucic
    # 
    # pomieszac kolejnosc tablicy attributes
    # 
    # wydzielic z nich 5 podzbiorow
    # 
    # SVM -libSVM 
    # wybrac nu-SVR - typ regresji
    # lub
    # epsilon-SVM
    # typ funkcji jadra RBF - optymalizowac parametr gamma
    # 
    # 
    # end of runWithParams()    

def main():
    gamma = None
    if(issetOption('-gamma')):
        gamma = float(getOption('-gamma'))

    nu = None
    if(issetOption('-nu')):
        nu = float(getOption('-nu'))

    runWithParams(nu, gamma, 1)

    nuValues = [0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.05, 0.03, 0.01, 0.008]
    gammaValues = [0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]

    # nuValues = [0.5, 0.008]
    # gammaValues = [0.8, 0.01]

    for nuV in nuValues:
        for gammaV in gammaValues:
            runWithParams(nuV, gammaV, 1)

# SETTINGS
global COLUMNS_COUNT, headers, attributes, columns, columnsNormalizationParams, lines, classes

# read and convert data


if(sys.argv.count('-i')):
    file = csv.reader(open(sys.argv[sys.argv.index('-i')+1], 'rb'), delimiter=',')
    loadDataFromOpenedFile();
    main();
# shuffle and write data
elif(sys.argv.count('-w')):
    global lines;
    lines = []
    for line in file:
        lines.append(line)    
    random.shuffle(lines)

    with open(sys.argv[sys.argv.index('-w')+1], 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for l in lines:
            spamwriter.writerow(l)
else:
    loadDataFromOpenedFile();
    main();

