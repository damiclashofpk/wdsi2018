# imports
# 
# We will use library 

# import pandas
# from pandas.plotting import scatter_matrix
# import matplotlib.pyplot as plt
# from sklearn import model_selection
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC

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

file = open("data.dat", "r") 


# pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose

def generateColumnsMatrix(columnsCount):
    columns = [];
    for i in range(columnsCount):
        columns.append([])
    return columns

def castToFloat(x):
    return float(x);

def findNormalisationDivider(maxV):
    powerOfTen = 0;
    while (maxV - (maxV % (10 ** powerOfTen))) / (10 ** powerOfTen) > 0:
        powerOfTen = powerOfTen + 1
    # print powerOfTen
    # print maxV
    return 10 ** powerOfTen


# SETTINGS

COLUMNS_COUNT = 8

#
# Attributes for estimation
headers = ["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
attributes = []

columns = generateColumnsMatrix(COLUMNS_COUNT)

# read and convert data

for line in file: 
    line = line.replace("\n", "").split(",")
    
    for key, item in enumerate(line[2:]):
        # print key
        columns[key].append(float(item));
    # columns 0 and 1 are useless for machine learning, so we ignore them
    currentAttribute = list(map(castToFloat, line[2:]))
    attributes.append(currentAttribute)
    # print currentAttribute
    
# normalise data

for idx, column in enumerate(columns):
    divider = findNormalisationDivider(max(column))
    for key, line in enumerate(attributes):
        attributes[key][idx] = attributes[key][idx] / divider;

# filtered data

for key, line in enumerate(attributes):
    print line

print 'count of records: '
print len(attributes)

##
#  Main Part
##

# Ostatnich dwócg wartości nie normalizujemy, a ostatnią mozna odrzucić
# 
# pomieszać kolejność tablicy attributes
# 
# wydzielić z nich 5 podzbiorów
# 
# SVM -libSVM 
# wybrać nu-SVR - typ regresji
# lub
# epsilon-SVM
# typ funkcji jądra RBF - optymalizować parametr gamma
# 
# 
