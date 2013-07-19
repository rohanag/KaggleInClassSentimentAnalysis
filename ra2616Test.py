from __future__ import division
from math import log
import sys
import json
import re
from nltk import stem
stemmer = stem.PorterStemmer()

def tokenizer(words):
    words = words.replace('\'','').replace(',','').replace('.','')
    temp = re.split("\s|(?<!\d)[^\w']+|[^\w']+(?!\d)", words.lower())
    return [stemmer.stem(i) for i in temp]

modelFile = sys.argv[1]
outFile = sys.argv[3]
test = open(sys.argv[2])
o = open(outFile,'w')
test.readline()
for i,data in enumerate(test):
    continue
n = i + 1
test = open(sys.argv[2])
test.readline()

with open(modelFile, 'r') as fp: 
    prob,sorted_features = json.load(fp)

num_features = len(prob)
y_pred = [0 for i in range(n)]
x = [ [0 for i in range(num_features)] for j in range(n) ]
for i,data in enumerate(test):
    datarow =  data[1:-1]
    tokenized = tokenizer(datarow)
    for token in tokenized:
        if token in sorted_features:
            x[i][sorted_features.index(token)] = 1
    for j in range(len(tokenized)-1):
        token = ' '.join((tokenized[j], tokenized[j+1]))
        if token in sorted_features:
            x[i][sorted_features.index(token)] = 1
            
for i in range(n):
    poslikhood = 0
    neglikhood = 0
    for j in range(num_features):
        if x[i][j] == 1:
            poslikhood += log(prob[j][1])
            neglikhood += log(prob[j][0])
        if x[i][j] == 0:
            poslikhood += log( 1 - prob[j][1] )
            neglikhood += log( 1 - prob[j][0] )
            
        if poslikhood > neglikhood:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    print >> o, y_pred[i]
o.close()
